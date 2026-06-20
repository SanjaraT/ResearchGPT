import json
import pandas as pd
import sys
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ----------------------------------------
# Path setup
# ----------------------------------------
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..")
    )
)

from app.rag.loader import load_all_pdfs
from app.rag.splitter import split_documents
from app.rag.embeddings import get_embedding_model
from app.rag.vectorstore import create_vector_store
from app.rag.query import build_qa_chain

# ----------------------------------------
# Sentence Transformer Model
# ----------------------------------------
print("Loading Sentence-BERT model...")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_similarity(text1, text2):
    emb1 = sbert_model.encode([text1])
    emb2 = sbert_model.encode([text2])

    return cosine_similarity(emb1, emb2)[0][0]

# ----------------------------------------
# Build RAG System
# ----------------------------------------

PDF_FOLDER = r"D:\A\DL\ResearchGPT\data\pdfs"

print("Loading PDFs...")
docs = load_all_pdfs(PDF_FOLDER)

print("Splitting documents...")
chunks = split_documents(docs)

print("Loading embeddings...")
embeddings = get_embedding_model()

print("Creating vectorstore...")
vectorstore = create_vector_store(chunks, embeddings)

print("Building QA chain...")
qa_chain, retriever = build_qa_chain(
    vectorstore
)


# ----------------------------------------
# Load Evaluation Dataset
# ----------------------------------------
with open("evaluation_set.json", "r", encoding="utf-8") as f:
    evaluation_data = json.load(f)

# ----------------------------------------
# Evaluation Loop
# ----------------------------------------

results = []
correct = 0

THRESHOLD = 0.70 

for item in evaluation_data:

    question = item["question"]
    expected = item["expected_answer"]

    response = qa_chain.invoke(
        {"question": question},
        config={
            "configurable": {
                "session_id": "evaluation"
            }
        }
    )

    # ----------------------------------------
    # Extract answer
    # ----------------------------------------
    if isinstance(response, dict):
        predicted = response.get("answer", "")
    else:
        predicted = str(response)

    # ----------------------------------------
    # Semantic similarity
    # ----------------------------------------
    similarity = get_similarity(expected, predicted)

    is_correct = similarity >= THRESHOLD

    if is_correct:
        correct += 1

    results.append({
        "Question": question,
        "Expected": expected,
        "Predicted": predicted,
        "Similarity": round(similarity, 4),
        "Correct": is_correct
    })

    print(
        f"{question[:60]}... "
        f"Similarity={similarity:.2f} "
        f"{'✓' if is_correct else '✗'}"
    )

# ----------------------------------------
# Final Score
# ----------------------------------------

accuracy = correct / len(evaluation_data)

print("\n" + "=" * 60)
print(f"Evaluation Accuracy: {accuracy:.2%}")
print("=" * 60)

# ----------------------------------------
# Save Results
# ----------------------------------------

df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)

print("\nResults saved to results.csv")