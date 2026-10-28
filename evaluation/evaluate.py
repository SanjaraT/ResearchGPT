"""Answer-level evaluation (calls the Groq LLM)."""
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
os.chdir(ROOT)  # the index path in vectorstore.py is relative to the repo root

from app.rag.embeddings import get_embedding_model
from app.rag.vectorstore import load_vector_store, get_vectorstore_statistics
from app.rag.query import build_qa_chain

THRESHOLD = 0.70
SWEEP = (0.60, 0.65, 0.70, 0.75, 0.80)

print("Loading Sentence-BERT model...")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_similarity(text1, text2):
    emb = sbert_model.encode([text1, text2])
    return float(cosine_similarity(emb[:1], emb[1:])[0][0])


print("Loading saved index...")
embeddings = get_embedding_model()
vectorstore = load_vector_store(embeddings)
print("Knowledge base:", get_vectorstore_statistics(vectorstore))
qa_chain, retriever = build_qa_chain(vectorstore)

evaluation_data = json.loads((HERE / "evaluation_set.json").read_text(encoding="utf-8"))
evaluation_data = list({x["question"]: x for x in evaluation_data}.values())
print(f"{len(evaluation_data)} unique questions")


def ask(question, session_id):
    """One fresh chat session per question; retry on rate limits."""
    for attempt in range(5):
        try:
            return qa_chain.invoke(
                {"question": question},
                config={"configurable": {"session_id": session_id}},
            )
        except Exception as e:
            wait = 10 * (attempt + 1)
            print(f"  retry {attempt + 1}/5 after error: {e} (waiting {wait}s)")
            time.sleep(wait)
    return ""


results = []
for i, item in enumerate(evaluation_data):
    question, expected = item["question"], item["expected_answer"]
    response = ask(question, session_id=f"eval-{i}")
    predicted = response.get("answer", "") if isinstance(response, dict) else str(response)

    similarity = get_similarity(expected, predicted)
    results.append({
        "Question": question,
        "Expected": expected,
        "Predicted": predicted,
        "Similarity": round(similarity, 4),
        "Correct": similarity >= THRESHOLD,
    })
    print(f"{question[:60]}... Similarity={similarity:.2f} {'OK' if similarity >= THRESHOLD else 'X'}")
    time.sleep(2)  # stay under the free-tier rate limit

n = len(results)
sims = [r["Similarity"] for r in results]
correct = sum(s >= THRESHOLD for s in sims)
refusals = sum("could not find" in r["Predicted"].lower() for r in results)

print("\n" + "=" * 60)
print(f"Mean similarity: {sum(sims) / n:.3f}")
print(f"Accuracy @ {THRESHOLD:.2f}: {correct}/{n} = {correct / n:.1%}")
print(f"Refusals ('could not find'): {refusals}")
print("Threshold sensitivity:")
for t in SWEEP:
    print(f"  >= {t:.2f}: {sum(s >= t for s in sims) / n:.1%}")
print("=" * 60)

pd.DataFrame(results).to_csv(HERE / "results.csv", index=False)
print("Saved evaluation/results.csv")


# import json
# import pandas as pd
# import sys
# import os
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer

# # ----------------------------------------
# # Path setup
# # ----------------------------------------
# sys.path.append(
#     os.path.abspath(
#         os.path.join(
#             os.path.dirname(__file__), "..")
#     )
# )

# from app.rag.loader import load_all_pdfs
# from app.rag.splitter import split_documents
# from app.rag.embeddings import get_embedding_model
# from app.rag.vectorstore import create_vector_store
# from app.rag.query import build_qa_chain

# # ----------------------------------------
# # Sentence Transformer Model
# # ----------------------------------------
# print("Loading Sentence-BERT model...")
# sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# def get_similarity(text1, text2):
#     emb1 = sbert_model.encode([text1])
#     emb2 = sbert_model.encode([text2])

#     return cosine_similarity(emb1, emb2)[0][0]

# # ----------------------------------------
# # Build RAG System
# # ----------------------------------------

# PDF_FOLDER = r"D:\A\DL\ResearchGPT\data\pdfs"

# print("Loading PDFs...")
# docs = load_all_pdfs(PDF_FOLDER)

# print("Splitting documents...")
# chunks = split_documents(docs)

# print("Loading embeddings...")
# embeddings = get_embedding_model()

# print("Creating vectorstore...")
# vectorstore = create_vector_store(chunks, embeddings)

# print("Building QA chain...")
# qa_chain, retriever = build_qa_chain(
#     vectorstore
# )


# # ----------------------------------------
# # Load Evaluation Dataset
# # ----------------------------------------
# with open("evaluation_set.json", "r", encoding="utf-8") as f:
#     evaluation_data = json.load(f)

# # ----------------------------------------
# # Evaluation Loop
# # ----------------------------------------

# results = []
# correct = 0

# THRESHOLD = 0.70 

# for item in evaluation_data:

#     question = item["question"]
#     expected = item["expected_answer"]

#     response = qa_chain.invoke(
#         {"question": question},
#         config={
#             "configurable": {
#                 "session_id": "evaluation"
#             }
#         }
#     )

#     # ----------------------------------------
#     # Extract answer
#     # ----------------------------------------
#     if isinstance(response, dict):
#         predicted = response.get("answer", "")
#     else:
#         predicted = str(response)

#     # ----------------------------------------
#     # Semantic similarity
#     # ----------------------------------------
#     similarity = get_similarity(expected, predicted)

#     is_correct = similarity >= THRESHOLD

#     if is_correct:
#         correct += 1

#     results.append({
#         "Question": question,
#         "Expected": expected,
#         "Predicted": predicted,
#         "Similarity": round(similarity, 4),
#         "Correct": is_correct
#     })

#     print(
#         f"{question[:60]}... "
#         f"Similarity={similarity:.2f} "
#         f"{'✓' if is_correct else '✗'}"
#     )

# # ----------------------------------------
# # Final Score
# # ----------------------------------------

# accuracy = correct / len(evaluation_data)

# print("\n" + "=" * 60)
# print(f"Evaluation Accuracy: {accuracy:.2%}")
# print("=" * 60)

# # ----------------------------------------
# # Save Results
# # ----------------------------------------

# df = pd.DataFrame(results)
# df.to_csv("results.csv", index=False)

# print("\nResults saved to results.csv")