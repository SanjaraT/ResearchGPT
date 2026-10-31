"""
Retrieval ablation (no LLM). Your original script, plus a row for the new
paper-aware retriever.

Run from the repo root:
    python evaluation/ablation.py
    python evaluation/ablation.py evaluation/evaluation_set_50_v5.json
"""
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
from app.rag.retriever import PaperAwareRetriever

vs = load_vector_store(get_embedding_model())
print("Knowledge base:", get_vectorstore_statistics(vs))

data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "evaluation" / "evaluation_set.json"
data = json.loads(data_path.read_text(encoding="utf-8"))
data = list({x["question"]: x for x in data}.values())  # drop verbatim duplicates

total = len(data)
data = [x for x in data if x.get("source_files") and "history" not in x]
print(f"Evaluating {len(data)} of {total} questions "
      f"(skipped {total - len(data)} not_found / follow-up items)")

K, FETCH_K, TOP = 10, 35, 5


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
    "MMR + rerank (old app)": (True, True,  None),
    "MMR + rerank + cap2":   (True,  True,   2),
}

# name -> function(question) -> top-5 documents
RETRIEVERS = {
    name: (lambda q, m=mmr, r=rerank, c=cap: retrieve(q, m, r, c))
    for name, (mmr, rerank, cap) in CONFIGS.items()
}
RETRIEVERS["paper-aware (new app)"] = PaperAwareRetriever(vectorstore=vs).retrieve

n = len(data)
n_multi = sum(len(x["source_files"]) > 1 for x in data)
n_single = n - n_multi
print(f"\n{'config':26s}{'Hit@5':>8s}{'Coverage@5':>12s}{'MRR':>8s}{'Multi-paper Cov':>17s}{'Gold chunks/5 (single)':>24s}")
for name, fn in RETRIEVERS.items():
    hit = cov = mrr = cov_multi = gold_chunks = 0.0
    for item in data:
        gold = set(item["source_files"])
        files = [d.metadata["source_file"] for d in fn(item["question"])]
        first = next((i for i, f in enumerate(files) if f in gold), None)
        if first is not None:
            hit += 1
            mrr += 1 / (first + 1)
        c = len(gold & set(files)) / len(gold)
        cov += c
        if len(gold) > 1:
            cov_multi += c
        else:
            gold_chunks += sum(f in gold for f in files)
    multi_str = f"{cov_multi / n_multi:.1%}" if n_multi else "n/a"
    single_str = f"{gold_chunks / n_single:.2f}" if n_single else "n/a"
    print(f"{name:26s}{hit/n:8.1%}{cov/n:12.1%}{mrr/n:8.3f}{multi_str:>17s}{single_str:>24s}")
