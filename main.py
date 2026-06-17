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


    # -----------------------------
    # Show Indexed Papers
    # -----------------------------
    if question.lower() == "papers":

        all_sources = set()

        docs = vectorstore.docstore._dict.values()

        for doc in docs:

            all_sources.add(
                doc.metadata["source_file"]
            )

        print("\nIndexed Papers:\n")

        for paper in sorted(all_sources):

            print("-", paper)

        continue

     # -----------------------------
    # Memory Command
    # ----------------------------
    if question.lower() == "history":
        print("\nConversation History:\n")

        for msg in qa_chain.memory.chat_memory.messages:
            print(msg)

        continue

    if question.lower() == "clear":

        qa_chain.memory.clear()

        print("Memory Cleared.")

        continue

    
    # -----------------------------
    # Compare Mode
    # -----------------------------
    if question.startswith("compare"):
        comparison_prompt = f"""
Compare the papers regarding:

{question}

Provide:

1. Objective
2. Dataset
3. Method
4. Results
5. Strengths
6. Limitations
"""
        result = qa_chain.invoke(
            {"question": comparison_prompt}
        )

        print("\nComparison:\n")
        print(result["answer"])

        continue

    # -----------------------------
    # Query
    # -----------------------------
    response = qa_chain.invoke(
        {"question": question}
    )

    print("\nAnswer:\n")

    print(response["answer"])

    print("\nSources:\n")

    seen = set()

    for doc in response["source_documents"]:
        source = doc.metadata.get(
            "source_file",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            "?"
        )

        citation = f"{source}(Page{page + 1})"

        if citation not in seen:
            print("-", citation)
            seen.add(citation)