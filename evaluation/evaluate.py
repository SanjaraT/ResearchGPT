import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # the index path in vectorstore.py is relative to the repo root

from dotenv import load_dotenv

load_dotenv()

from evaluation.scoring import (
    JUDGE_PROMPT,
    SIM_SWEEP,
    SIM_THRESHOLD,
    grade,
    parse_verdict,
    similarity,
)

COLUMNS = [
    "ID", "Category", "Question", "Expected", "Predicted", "Similarity",
    "SimCorrect", "Verdict", "NumbersOK", "MissingNumbers", "Correct",
    "Words", "RetrievedFiles", "SourceHit", "Reason", "HistoryTag",
]

_BOOL_COLUMNS = ("SimCorrect", "NumbersOK", "Correct")


class Paused(Exception):
    """The run cannot continue right now (rate limit, API error). Resume later."""


# ----------------------------------------------------------------------
# results file: load / save / keys
# ----------------------------------------------------------------------

def history_tag(history):
    """Short fingerprint of a follow-up's history ('' when there is none)."""

    if not history:
        return ""

    return hashlib.sha1(
        json.dumps(history, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]


def row_key(question, history=None):

    return (question, history_tag(history))


def load_rows(path):
    """Saved rows keyed by (question, history tag), with real types restored."""

    path = Path(path)

    if not path.exists():
        return {}

    with open(path, newline="", encoding="utf-8") as f:

        reader = csv.DictReader(f)

        if set(COLUMNS) - set(reader.fieldnames or []):

            print(
                f"  {path.name} was written by an older version of this script "
                f"(missing columns); ignoring it."
            )

            return {}

        rows = {}

        for row in reader:

            for column in _BOOL_COLUMNS:
                row[column] = row[column] == "True"

            row["Similarity"] = float(row["Similarity"] or 0)
            row["Words"] = int(row["Words"] or 0)
            row["SourceHit"] = float(row["SourceHit"]) if row["SourceHit"] != "" else ""

            if str(row["ID"]).isdigit():
                row["ID"] = int(row["ID"])

            rows[(row["Question"], row["HistoryTag"])] = row

    return rows


def save_rows(path, rows):
    """Atomic write: a crash mid-write leaves the previous file intact."""

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")

    with open(tmp, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    for attempt in range(5):

        try:

            os.replace(tmp, path)

            return

        except PermissionError:  # the CSV is open in Excel

            print(f"  cannot write {path.name}: close it in Excel (retry {attempt + 1}/5)")

            time.sleep(3)

    raise PermissionError(
        f"Could not write {path}. Close it in Excel and run the same command again; "
        f"your progress is kept in {tmp.name}."
    )


# ----------------------------------------------------------------------
# LLM calls
# ----------------------------------------------------------------------

def make_judge():

    from langchain_groq import ChatGroq

    from app.rag.config import JUDGE_MODEL

    llm = ChatGroq(
        model=JUDGE_MODEL,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
        reasoning_effort="medium",
        max_tokens=2048,
    )

    def judge(question, expected, predicted):

        prompt = (
            JUDGE_PROMPT
            .replace("{question}", question)
            .replace("{expected}", expected)
            .replace("{predicted}", predicted)
        )

        for attempt in range(3):

            try:

                return parse_verdict(llm.invoke(prompt).content)

            except Exception as error:  # rate limit / transient API error

                print(f"  judge retry {attempt + 1}/3: {error}")

                time.sleep(5 * (attempt + 1))

        return "error", "judge call failed"

    return judge


def ask(chain, question, session_id):
    """Invoke the chain; retry with backoff, then pause the run (never return a fake empty answer)."""

    last_error = None

    for attempt in range(5):

        try:

            response = chain.invoke(
                {"question": question},
                config={"configurable": {"session_id": session_id}},
            )

            return (
                response.get("answer", "")
                if isinstance(response, dict)
                else str(response)
            )

        except Exception as error:

            last_error = error

            wait = 10 * (attempt + 1)

            print(f"  retry {attempt + 1}/5 after error: {error} (waiting {wait}s)")

            time.sleep(wait)

    raise Paused(f"answer generation failed after 5 attempts: {last_error}")


# ----------------------------------------------------------------------
# the evaluation loop (resumable)
# ----------------------------------------------------------------------

def source_hit(gold_files, retrieved_files):
    """Fraction of the gold papers that were retrieved ('' if no gold)."""

    if not gold_files:
        return ""

    return round(len(set(gold_files) & set(retrieved_files)) / len(gold_files), 2)


def run(
    data,
    out_path,
    generate,
    retrieve_files,
    judge,
    sim_fn,
    prepare_history,
    sleep=2.0,
    fresh=False,
    log=print,
):
    """
    Returns dict(rows, generated, regraded, skipped, paused).
    rows = results for the questions in `data`, in order; paused = reason or None.
    """

    results = {} if fresh else load_rows(out_path)

    keys = [row_key(x["question"], x.get("history")) for x in data]

    def persist():

        in_set = [results[k] for k in keys if k in results]
        others = [r for k, r in results.items() if k not in set(keys)]

        save_rows(out_path, in_set + others)

    counts = {"generated": 0, "regraded": 0, "skipped": 0}
    paused = None

    try:

        for index, (item, key) in enumerate(zip(data, keys), start=1):

            question = item["question"]
            expected = item["expected_answer"]
            history = item.get("history", [])
            category = item.get("category", "all")
            tag = f"[{index:>3}/{len(data)}] {category:<10}"

            prev = results.get(key)

            if prev and prev["Verdict"] != "error" and prev["Expected"] == expected:

                counts["skipped"] += 1

                log(f"{tag} done earlier, skipped  {question[:50]}")

                continue

            if prev:

                # answer already saved: only grade again, do not generate
                predicted = prev["Predicted"]
                retrieved_text = prev["RetrievedFiles"]
                hit = prev["SourceHit"]
                counts["regraded"] += 1

            else:

                session_id = f"eval-{index}"

                prepare_history(session_id, history)

                predicted = generate(question, session_id)

                if history:  # follow-ups need the rewritten question: no retrieval diagnostics
                    retrieved_text, hit = "", ""
                else:
                    retrieved = retrieve_files(question)
                    retrieved_text = " | ".join(retrieved)
                    hit = source_hit(item.get("source_files", []), retrieved)

                counts["generated"] += 1

                # save the answer before grading, so a failure while grading
                # never costs another generation
                results[key] = {
                    "ID": item.get("id", index), "Category": category,
                    "Question": question, "Expected": expected,
                    "Predicted": predicted, "Similarity": 0.0, "SimCorrect": False,
                    "Verdict": "error", "NumbersOK": False, "MissingNumbers": "",
                    "Correct": False, "Words": len(predicted.split()),
                    "RetrievedFiles": retrieved_text, "SourceHit": hit,
                    "Reason": "not graded yet", "HistoryTag": history_tag(history),
                }

                persist()

            result = grade(
                question,
                expected,
                predicted,
                judge,
                strict_numbers=category in ("direct", "follow_up", "all"),
            )

            sim = sim_fn(expected, predicted)

            results[key] = {
                "ID": item.get("id", index),
                "Category": category,
                "Question": question,
                "Expected": expected,
                "Predicted": predicted,
                "Similarity": round(sim, 4),
                "SimCorrect": sim >= SIM_THRESHOLD,
                "Verdict": result["verdict"],
                "NumbersOK": result["numbers_ok"],
                "MissingNumbers": result["missing_numbers"],
                "Correct": result["correct"],
                "Words": len(predicted.split()),
                "RetrievedFiles": retrieved_text,
                "SourceHit": hit,
                "Reason": result["reason"],
                "HistoryTag": history_tag(history),
            }

            persist()

            log(
                f"{tag} {result['verdict']:<8} numbers_ok={result['numbers_ok']!s:<5} "
                f"sim={sim:.2f} {question[:50]}"
            )

            if result["verdict"] == "error":
                raise Paused("the judge call failed (rate limit?); the answer is saved")

            if not prev:
                time.sleep(sleep)

    except Paused as reason:

        paused = str(reason)

    except KeyboardInterrupt:

        paused = "stopped by you (Ctrl+C)"

    return {
        "rows": [results[k] for k in keys if k in results],
        "paused": paused,
        **counts,
    }


# ----------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------

def summarize(rows):

    groups = defaultdict(list)

    for row in rows:
        groups[row["Category"]].append(row)
        groups["ALL"].append(row)

    print("\n" + "=" * 72)
    print(f"{'category':<12}{'n':>4}{'correct':>9}{'partial':>9}{'wrong':>7}{'strict':>9}{'lenient':>9}")

    for name in [c for c in groups if c != "ALL"] + ["ALL"]:

        items = groups[name]
        n = len(items)
        correct = sum(r["Correct"] for r in items)
        partial = sum(r["Verdict"] == "partial" for r in items)
        wrong = n - correct - partial

        print(
            f"{name:<12}{n:>4}{correct:>9}{partial:>9}{wrong:>7}"
            f"{correct / n:>9.0%}{(correct + 0.5 * partial) / n:>9.0%}"
        )

    print("=" * 72)
    print("strict = correct only; lenient = partial answers count half")

    fallback = "could not find the answer"
    refused = [r for r in rows if fallback in r["Predicted"].lower()]
    wrongly = [r for r in refused if r["Category"] != "not_found"]
    print(
        f"refusals ('could not find'): {len(refused)} "
        f"({len(wrongly)} on answerable questions)"
    )

    # legacy embedding-similarity score, for comparison with old runs
    sims = [r["Similarity"] for r in rows]
    n = len(rows)
    print("\nlegacy similarity score (all-MiniLM-L6-v2, same as the old runs)")
    print(f"  mean similarity: {sum(sims) / n:.3f}")
    print(f"  accuracy @ {SIM_THRESHOLD:.2f}: {sum(r['SimCorrect'] for r in rows)}/{n} = "
          f"{sum(r['SimCorrect'] for r in rows) / n:.1%}")
    print("  threshold sensitivity: " + "  ".join(
        f">={t:.2f}: {sum(s >= t for s in sims) / n:.0%}" for t in SIM_SWEEP))
    disagree = [r for r in rows if r["SimCorrect"] != r["Correct"]]
    print(f"  similarity and judge disagree on {len(disagree)}/{n} questions: "
          + ", ".join(f"id {r['ID']}" for r in disagree))

    judged = [r for r in rows if r["SourceHit"] != ""]

    if judged:

        missed = [r for r in judged if r["SourceHit"] < 1]

        print(
            f"\nretrieval: gold paper(s) fully retrieved for "
            f"{len(judged) - len(missed)}/{len(judged)} questions"
        )

        for r in missed:
            print(f"  missed  id={r['ID']}  got: {r['RetrievedFiles']}")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default=str(ROOT / "evaluation" / "evaluation_set.json"))
    parser.add_argument("--out", default=str(ROOT / "evaluation" / "results_judged.csv"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=2.0, help="seconds between questions (rate limits)")
    parser.add_argument("--fresh", action="store_true", help="ignore the saved results and start over")
    args = parser.parse_args()

    from app.rag.embeddings import get_embedding_model
    from app.rag.vectorstore import get_vectorstore_statistics, load_vector_store
    from app.rag.query import build_qa_chain, get_session_history_public

    print("Loading index...")
    vectorstore = load_vector_store(get_embedding_model())
    print("Knowledge base:", get_vectorstore_statistics(vectorstore))

    chain, retriever = build_qa_chain(vectorstore)
    judge = make_judge()

    from sentence_transformers import SentenceTransformer

    print("Loading Sentence-BERT model (legacy similarity score)...")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    data = json.loads(Path(args.set).read_text(encoding="utf-8"))

    if args.limit:
        data = data[: args.limit]

    def prepare_history(session_id, history):

        memory = get_session_history_public(session_id)

        for turn in history:

            if turn["role"] == "user":
                memory.add_user_message(turn["content"])
            else:
                memory.add_ai_message(turn["content"])

    outcome = run(
        data,
        args.out,
        generate=lambda question, session_id: ask(chain, question, session_id),
        retrieve_files=lambda q: sorted({d.metadata["source_file"] for d in retriever.retrieve(q)}),
        judge=judge,
        sim_fn=lambda expected, predicted: similarity(sbert, expected, predicted),
        prepare_history=prepare_history,
        sleep=args.sleep,
        fresh=args.fresh,
    )

    rows = outcome["rows"]

    print(
        f"\nthis run: {outcome['generated']} generated, {outcome['regraded']} re-graded, "
        f"{outcome['skipped']} skipped (already done)"
    )

    if rows:
        summarize(rows)

    print(f"\nResults saved to {args.out}")

    if outcome["paused"]:

        print(
            f"\nPAUSED: {outcome['paused']}\n"
            f"{len(rows)} of {len(data)} questions are saved. Wait a few minutes "
            f"(a daily limit may need longer) and run the same command again to continue."
        )

        sys.exit(2)


if __name__ == "__main__":
    main()