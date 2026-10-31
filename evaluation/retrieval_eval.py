import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def score_config(data, retrieve_fn):
    """retrieve_fn(question) -> list of Documents (top 5)."""

    n = len(data)
    n_multi = sum(len(x["source_files"]) > 1 for x in data)
    n_single = n - n_multi

    hit = cov = mrr = cov_multi = gold_chunks = 0.0

    for item in data:

        gold = set(item["source_files"])

        files = [d.metadata["source_file"] for d in retrieve_fn(item["question"])]

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

    return {
        "Hit@5": hit / n,
        "Coverage@5": cov / n,
        "MRR": mrr / n,
        "Multi-paper Cov": cov_multi / n_multi if n_multi else None,
        "Gold chunks/5": gold_chunks / n_single if n_single else None,
    }


def select_questions(data):

    data = list({x["question"]: x for x in data}.values())  # drop verbatim duplicates

    kept = [x for x in data if x.get("source_files") and "history" not in x]

    return kept, len(data) - len(kept)


def print_table(results):

    cols = ["Hit@5", "Coverage@5", "MRR", "Multi-paper Cov", "Gold chunks/5"]

    print(f"\n{'config':28s}" + "".join(f"{c:>17s}" for c in cols))

    for name, scores in results.items():

        cells = []

        for c in cols:

            v = scores[c]

            if v is None:
                cells.append("n/a")
            elif c in ("MRR", "Gold chunks/5"):
                cells.append(f"{v:.2f}")
            else:
                cells.append(f"{v:.1%}")

        print(f"{name:28s}" + "".join(f"{c:>17s}" for c in cells))


def main():

    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)  # the index path in vectorstore.py is relative to the repo root

    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default=str(ROOT / "evaluation" / "evaluation_set.json"))
    args = parser.parse_args()

    from app.rag.embeddings import get_embedding_model
    from app.rag.reranker import rerank_documents
    from app.rag.retriever import PaperAwareRetriever
    from app.rag.vectorstore import get_vectorstore_statistics, load_vector_store

    vs = load_vector_store(get_embedding_model())
    print("Knowledge base:", get_vectorstore_statistics(vs))

    data = json.loads(Path(args.set).read_text(encoding="utf-8"))
    data, skipped = select_questions(data)
    print(f"Evaluating {len(data)} questions (skipped {skipped} not_found / follow-up items)")

    retriever = PaperAwareRetriever(vectorstore=vs)

    configs = {
        "dense top-5": lambda q: vs.similarity_search(q, k=5),
        "MMR + rerank (old app)": lambda q: rerank_documents(
            q, vs.max_marginal_relevance_search(q, k=10, fetch_k=35), top_k=5
        ),
        "paper-aware (new app)": retriever.retrieve,
    }

    print_table({name: score_config(data, fn) for name, fn in configs.items()})


if __name__ == "__main__":
    main()
