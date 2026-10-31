"""
Answer grading for the evaluation set.

Why not cosine similarity: embeddings barely react to a changed number or
entity ("20 convolutional layers" scores 0.76 against "five convolutional
layers"), and they penalise correct answers that are worded differently or are
longer than the reference. Grading here is:

  1. not_found questions  -> exact match on the fallback sentence
  2. everything else      -> LLM judge (correct / partial / wrong)
                             AND every number in the reference must appear in the answer
"""

import json
import re
import unicodedata

FALLBACK = "I could not find the answer in the provided papers."

# Legacy embedding-similarity score, kept so new runs can be compared with old ones.
SIM_THRESHOLD = 0.70
SIM_SWEEP = (0.60, 0.65, 0.70, 0.75, 0.80)

JUDGE_PROMPT = """You are grading an answer from a research-paper question answering system.

Question: {question}
Reference answer: {expected}
System answer: {predicted}

Grade the system answer against the reference answer:
- "correct": it states the key facts of the reference answer. Wording may differ and extra correct detail is fine.
- "partial": it states some but not all of the key facts, or it is vague or only repeats the question.
- "wrong": it contradicts the reference, gives different numbers, names or entities, refuses although the reference has an answer, or is empty.
Differences in numbers, names or entities are errors even if the wording is similar.

Reply with JSON only: {"verdict": "correct" | "partial" | "wrong", "reason": "<one short sentence>"}"""

# "one" is left out on purpose: it is mostly a pronoun / article ("one evaluation")
_NUMBER_WORDS = {
    "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

# digits glued to a letter or hyphen are part of a name ("GPT-3", "T5",
# "LLaMA-13B", "ILSVRC-2012") and are not compared; the digit lookbehind stops
# a match from starting in the middle of a longer number
_NUMBER = re.compile(r"(?<![A-Za-z0-9\-])\d[\d,]*(?:\.\d+)?")


def normalize_text(text):
    """NFKC + unify dashes and quotes; models often emit U+2011 / U+202F."""

    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub("[‐-―−]", "-", text)
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')

    return re.sub(r"\s+", " ", text).strip()


def extract_numbers(text):

    text = normalize_text(text).lower()

    for word, digit in _NUMBER_WORDS.items():
        text = re.sub(rf"\b{word}\b", digit, text)

    return {
        match.replace(",", "").rstrip(".")
        for match in _NUMBER.findall(text)
    }


def parse_verdict(text):
    """Pull the verdict out of the judge's reply. Returns (verdict, reason)."""

    match = re.search(r"\{.*\}", text or "", re.S)

    if match:

        try:

            data = json.loads(match.group(0))

            verdict = str(data.get("verdict", "")).strip().lower()

            if verdict in {"correct", "partial", "wrong"}:

                return verdict, str(data.get("reason", ""))[:200]

        except json.JSONDecodeError:
            pass

    return "error", (text or "")[:120]


def _is_refusal(text):

    return normalize_text(text).strip('"') == FALLBACK


def grade(question, expected, predicted, judge, strict_numbers=True):
    """
    judge(question, expected, predicted) -> (verdict, reason)

    strict_numbers: every number in the reference must appear in the answer.
    Use it for short factual references; for summaries and comparisons the
    numbers are still reported but do not decide `correct`, because a 30-50
    word answer cannot repeat every figure of the reference.

    Returns dict(verdict, numbers_ok, missing_numbers, correct, reason).
    """

    expected = normalize_text(expected)
    predicted = normalize_text(predicted)

    def result(verdict, reason, missing=()):

        numbers_ok = not missing

        return {
            "verdict": verdict,
            "numbers_ok": numbers_ok,
            "missing_numbers": " ".join(sorted(missing)),
            "correct": verdict == "correct" and (numbers_ok or not strict_numbers),
            "reason": reason,
        }

    if _is_refusal(expected):

        if _is_refusal(predicted):
            return result("correct", "exact fallback sentence")

        return result("wrong", "answered a question the papers do not cover")

    if not predicted:
        return result("wrong", "empty answer")

    if _is_refusal(predicted):
        return result("wrong", "refused although the papers cover the question")

    verdict, reason = judge(question, expected, predicted)

    missing = extract_numbers(expected) - extract_numbers(predicted)

    return result(verdict, reason, missing)


def similarity(model, expected, predicted):
    """Cosine similarity of two texts with a SentenceTransformer-like model (raw text, as in the old runs)."""

    import numpy as np

    emb = model.encode([expected, predicted])

    a, b = np.asarray(emb[0], dtype=float), np.asarray(emb[1], dtype=float)

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    return float(a @ b / denom) if denom else 0.0
