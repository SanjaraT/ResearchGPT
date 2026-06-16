from app.rag.loader import load_all_pdfs
from app.rag.splitter import split_documents
from app.rag.embeddings import get_embedding_model
from app.rag.vectorstore import (
    create_vector_store,
    save_vector_store
)
from app.rag.query import build_qa_chain


PDF_FOLDER = r"D:\A\DL\ResearchGPT\data\pdfs"


# --------------------------------------------------
# Load PDFs
# --------------------------------------------------

print("\nLoading PDFs...")

docs = load_all_pdfs(PDF_FOLDER)

print(f"Loaded Pages: {len(docs)}")


# --------------------------------------------------
# Chunking
# --------------------------------------------------

print("\nSplitting Documents...")

chunks = split_documents(docs)

print(f"Total Chunks: {len(chunks)}")


# --------------------------------------------------
# Embeddings
# --------------------------------------------------

print("\nLoading Embedding Model...")

embeddings = get_embedding_model()


# --------------------------------------------------
# Vector Store
# --------------------------------------------------

print("\nCreating FAISS Vector Store...")

vectorstore = create_vector_store(
    chunks,
    embeddings
)

save_vector_store(vectorstore)

print("FAISS Index Saved")


# --------------------------------------------------
# QA Chain
# --------------------------------------------------

print("\nBuilding ResearchGPT QA Chain...")

qa_chain = build_qa_chain(
    vectorstore
)

print("\nResearchGPT Ready!")
print("Type 'exit' to quit.\n")


# --------------------------------------------------
# Chat Loop
# --------------------------------------------------

while True:

    question = input("\nAsk Question: ")

    if question.lower() == "exit":
        break

    response = qa_chain.invoke(
        {"query": question}
    )

    print("\nAnswer:\n")

    print(response["result"])