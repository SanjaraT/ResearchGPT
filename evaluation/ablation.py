"""
Retrieval ablation table (no LLM, no API key): "does the right paper reach the top 5?"

For every question that has gold `source_files` it records which papers the
retriever returns in its top 5 chunks and prints a Markdown table:

| Config | Hit@5 | Coverage@5 | MRR | Multi-paper Coverage |

  Hit@5                 at least one gold paper is among the top 5 chunks
  Coverage@5            fraction of the gold papers found in the top 5
  MRR                   1 / rank of the first chunk from a gold paper
  Multi-paper Coverage  Coverage@5 on questions that need more than one paper

Rows:
  dense top-5                       plain vector similarity
  MMR top-5                         maximal marginal relevance (k=10, fetch_k=35, first 5)
  dense + rerank                    dense top-10, cross-encoder, keep 5
  MMR + rerank (old app)            what the app did before the fixes
  MMR + rerank + cap of 2 chunks/paper (experiment)
  paper-aware routing (new app)     what the app does now

Besides the full table it prints a second one for the questions that do NOT
name any paper ("unnamed"), the cases where routing cannot help and the
retriever has to find the paper from the content alone.

Not counted: not_found questions (no gold paper) and follow-ups (they need the
chat history rewritten first).

Run from anywhere:

    python evaluation/ablation.py
    python evaluation/ablation.py evaluation/evaluation_set_50_v5.json      (old style: set as first argument)
    python evaluation/ablation.py --set evaluation/evaluation_set_110.json
    python evaluation/ablation.py --out evaluation/ablation_results.md

The Markdown tables are saved to --out. The console also shows the older
table with the extra "Gold chunks/5" column (how many of the 5 chunks come
from the gold paper, single-paper questions only).
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.retrieval_eval import score_config, select_questions

K, FETCH_K, TOP, CAP = 10, 35, 5, 2

HEADER = "| Config | Hit@5 | Coverage@5 | MRR | Multi-paper Coverage |"
SEPARATOR = "|---|---|---|---|---|"


def cap_per_source(docs, max_per, top):
    """Keep the ranking order but allow at most max_per chunks from one paper."""

    out, counts = [], {}

    for doc in docs:

        source = doc.metadata["source_file"]

        if counts.get(source, 0) < max_per:
            out.append(doc)
            counts[source] = counts.get(source, 0) + 1

        if len(out) == top:
            break

    return out


def split_by_naming(data, detect_papers, available=None):
    """(named, unnamed): does the question mention any paper by title or alias?"""

    named, unnamed = [], []

    for item in data:

        target = named if detect_papers(item["question"], available) else unnamed
        target.append(item)

    return named, unnamed


def format_table(results):
    """Markdown table; `results` maps config name -> score_config() output."""

    def pct(value):
        return "n/a" if value is None else f"{value:.1%}"

    lines = [HEADER, SEPARATOR]

    for name, s in results.items():

        lines.append(
            f"| {name} | {pct(s['Hit@5'])} | {pct(s['Coverage@5'])} "
            f"| {s['MRR']:.3f} | {pct(s['Multi-paper Cov'])} |"
        )

    return "\n".join(lines)


def build_retrievers(vectorstore, rerank_documents, paper_aware_retrieve):
    """
    name -> function(question) -> top-5 documents.

    The candidate lists are computed once per question and shared between the
    rows that start from the same candidates, so the cross-encoder runs twice
    per question instead of four times.
    """

    cache = {}

    def dense_candidates(q):
        key = ("dense", q)
        if key not in cache:
            cache[key] = vectorstore.similarity_search(q, k=K)
        return cache[key]

    def mmr_candidates(q):
        key = ("mmr", q)
        if key not in cache:
            cache[key] = vectorstore.max_marginal_relevance_search(q, k=K, fetch_k=FETCH_K)
        return cache[key]

    def reranked(kind, q, docs):
        key = ("rerank", kind, q)
        if key not in cache:
            cache[key] = rerank_documents(q, docs, top_k=len(docs))
        return cache[key]

    return {
        "dense top-5": lambda q: dense_candidates(q)[:TOP],
        # MMR picks greedily, so its first 5 picks equal an MMR search with k=5
        "MMR top-5": lambda q: mmr_candidates(q)[:TOP],
        "dense + rerank": lambda q: reranked("dense", q, dense_candidates(q))[:TOP],
        "MMR + rerank (old app)": lambda q: reranked("mmr", q, mmr_candidates(q))[:TOP],
        f"MMR + rerank + cap of {CAP} chunks/paper (experiment)": lambda q: cap_per_source(
            reranked("mmr", q, mmr_candidates(q)), CAP, TOP
        ),
        "paper-aware routing (new app)": paper_aware_retrieve,
    }


def main():

    os.chdir(ROOT)  # the index path in vectorstore.py is relative to the repo root

    parser = argparse.ArgumentParser()
    parser.add_argument("set_path", nargs="?", help="question set (same as --set)")
    parser.add_argument("--set", dest="set_opt")
    parser.add_argument("--out", default=str(ROOT / "evaluation" / "ablation_results.md"))
    args = parser.parse_args()
    args.set = args.set_opt or args.set_path or str(ROOT / "evaluation" / "evaluation_set100.json")

    from evaluation.retrieval_eval import print_table
    from app.rag.embeddings import get_embedding_model
    from app.rag.papers import detect_papers
    from app.rag.reranker import rerank_documents
    from app.rag.retriever import PaperAwareRetriever
    from app.rag.vectorstore import (
        get_existing_papers,
        get_vectorstore_statistics,
        load_vector_store,
    )

    vs = load_vector_store(get_embedding_model())
    print("Knowledge base:", get_vectorstore_statistics(vs))

    data = json.loads(Path(args.set).read_text(encoding="utf-8"))
    data, skipped = select_questions(data)

    indexed = get_existing_papers(vs)
    gold = {f for x in data for f in x["source_files"]}

    if gold - indexed:
        print(
            "WARNING: these gold papers are not in the index, so their questions "
            "count as misses: " + ", ".join(sorted(gold - indexed))
            + "\n         Rebuild the index (python create_index.py) with all PDFs."
        )

    named, unnamed = split_by_naming(data, detect_papers, available=indexed)

    print(
        f"Evaluating {len(data)} questions "
        f"({len(named)} name a paper, {len(unnamed)} do not; "
        f"skipped {skipped} not_found / follow-up items)\n"
    )

    retrievers = build_retrievers(
        vs, rerank_documents, PaperAwareRetriever(vectorstore=vs).retrieve
    )

    sections = []
    first_results = None

    for title, subset in (
        (f"All questions with a gold paper (n={len(data)})", data),
        (f"Questions that name no paper (n={len(unnamed)})", unnamed),
    ):

        if not subset:
            continue

        results = {}

        for name, fn in retrievers.items():

            print(f"  [{len(subset)} questions] {name}")

            results[name] = score_config(subset, fn)

        first_results = first_results or results

        sections.append(f"### {title}\n\n{format_table(results)}")

    report = "\n\n".join(sections)

    print("\n" + report + "\n")

    print("Same numbers with the extra column (all questions):")
    print_table(first_results)

    Path(args.out).write_text(report + "\n", encoding="utf-8")

    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()