import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
os.chdir(ROOT)  # the index path in vectorstore.py is relative to the repo root

from app.rag.embeddings import get_embedding_model
from app.rag.vectorstore import load_vector_store, get_vectorstore_statistics
from app.rag.reranker import rerank_documents

vs = load_vector_store(get_embedding_model())
print("Knowledge base:", get_vectorstore_statistics(vs))

data = json.loads((ROOT / "evaluation" / "evaluation_set.json").read_text(encoding="utf-8"))
data = list({x["question"]: x for x in data}.values())  # drop verbatim duplicates
if any(not x.get("source_files") for x in data):
    sys.exit("Some questions have no source_files. Run evaluation/label_sources.py first.")

K, FETCH_K, TOP = 10, 35, 5  # same values as app/rag/query.py

# Files that count as valid evidence for the same gold label.
# EDIT: the key must match the GPT string in your evaluation_set.json exactly.
FAMILY = {"GPT-1.pdf": {"GPT-1.pdf", "Language_model.pdf", "Few_Shot.pdf"}}


def acceptable(gold_file):
    return FAMILY.get(gold_file, {gold_file})


def cap_per_source(docs, max_per, top):
    """Keep ranking order but allow at most max_per chunks from any one paper."""
    out, counts = [], {}
    for d in docs:
        s = d.metadata["source_file"]
        if counts.get(s, 0) < max_per:
            out.append(d)
            counts[s] = counts.get(s, 0) + 1
        if len(out) == top:
            break
    return out


def retrieve(question, mmr, rerank, cap=None):
    retriever = vs.as_retriever(
        search_type="mmr" if mmr else "similarity",
        search_kwargs={"k": K, **({"fetch_k": FETCH_K} if mmr else {})},
    )
    docs = retriever.invoke(question)
    if not rerank:
        return docs[:TOP]
    if cap is None:
        return rerank_documents(question, docs, top_k=TOP)
    ranked = rerank_documents(question, docs, top_k=len(docs))
    return cap_per_source(ranked, cap, TOP)


CONFIGS = {
    "dense top-5":           (False, False, None),
    "MMR top-5":             (True,  False, None),
    "dense + rerank":        (False, True,  None),
    "MMR + rerank (app)":    (True,  True,   None),
    "MMR + rerank + cap2":   (True,  True,   2),
}

n = len(data)
n_multi = sum(len(x["source_files"]) > 1 for x in data)
print(f"\n{'config':24s}{'Hit@5':>8s}{'Coverage@5':>12s}{'MRR':>8s}{'Multi-paper Cov':>17s}")
for name, (mmr, rerank, cap) in CONFIGS.items():
    hit = cov = mrr = cov_multi = 0.0
    for item in data:
        gold = item["source_files"]
        files = [d.metadata["source_file"] for d in retrieve(item["question"], mmr, rerank, cap)]
        first = next((i for i, f in enumerate(files) if any(f in acceptable(g) for g in gold)), None)
        if first is not None:
            hit += 1
            mrr += 1 / (first + 1)
        c = sum(bool(acceptable(g) & set(files)) for g in gold) / len(gold)
        cov += c
        if len(gold) > 1:
            cov_multi += c
    print(f"{name:24s}{hit/n:8.1%}{cov/n:12.1%}{mrr/n:8.3f}{cov_multi/n_multi:17.1%}")