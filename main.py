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


# --------------------------------------------------
# Build QA Chain
# --------------------------------------------------

print("\nBuilding ResearchGPT QA Chain...")

qa_chain, retriever = build_qa_chain(
    vectorstore
)

print("\nResearchGPT Ready!")
# print("\nCommands:")
# print("papers   -> list indexed papers")
# print("history  -> show conversation memory")
# print("clear    -> clear memory")
# print("stats    -> show index statistics")
# print("compare <topic>")
print("Type exit to end the session!\n")


# --------------------------------------------------
# Session Memory
# --------------------------------------------------

SESSION_ID = "user_1"


# --------------------------------------------------
# Chat Loop
# --------------------------------------------------

while True:

    question = input("\nAsk Question: ").strip()

    if not question:
        continue

    if question.lower() == "exit":
        break

    # --------------------------------------------------
    # Papers Command
    # --------------------------------------------------

    if question.lower() == "papers":

        all_sources = set()

        docs_in_store = vectorstore.docstore._dict.values()

        for doc in docs_in_store:

            source = doc.metadata.get(
                "source_file",
                "Unknown"
            )

            all_sources.add(source)

        print("\nIndexed Papers:\n")

        for paper in sorted(all_sources):

            print("-", paper)

        continue

    # --------------------------------------------------
    # History Command
    # --------------------------------------------------

    if question.lower() == "history":

        history = qa_chain.get_session_history(
            SESSION_ID
        )

        print("\nConversation History:\n")

        if len(history.messages) == 0:

            print("No conversation history found.")

        else:

            for msg in history.messages:
                print(msg)

        continue

    # --------------------------------------------------
    # Clear Memory
    # --------------------------------------------------

    if question.lower() == "clear":

        qa_chain.get_session_history(
            SESSION_ID
        ).clear()

        print("\nMemory Cleared.")

        continue

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    if question.lower() == "stats":

        total_docs = len(
            set(
                doc.metadata.get(
                    "source_file",
                    "Unknown"
                )
                for doc in vectorstore.docstore._dict.values()
            )
        )

        total_chunks = len(
            vectorstore.docstore._dict
        )

        print("\nResearchGPT Statistics\n")

        print(f"Indexed Papers : {total_docs}")
        print(f"Total Chunks   : {total_chunks}")

        continue

    # --------------------------------------------------
    # Compare Mode
    # --------------------------------------------------

    if question.lower().startswith("compare"):

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
            {"question": comparison_prompt},
            config={
                "configurable": {
                    "session_id": SESSION_ID
                }
            }
        )

        print("\nComparison:\n")
        print(result)

        continue

    # --------------------------------------------------
    # Retrieve Sources
    # --------------------------------------------------

    docs = retriever.invoke(question)

    # --------------------------------------------------
    # Ask RAG Chain
    # --------------------------------------------------

    response = qa_chain.invoke(
        {"question": question},
        config={
            "configurable": {
                "session_id": SESSION_ID
            }
        }
    )

    print("\nAnswer:\n")

    print(response)

    # --------------------------------------------------
    # Sources
    # --------------------------------------------------

    print("\nSources:\n")

    seen = set()

    for doc in docs:

        source = doc.metadata.get(
            "source_file",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            0
        )

        citation = (
            f"{source} "
            f"(Page {page + 1})"
        )

        if citation not in seen:

            print("-", citation)

            seen.add(citation)