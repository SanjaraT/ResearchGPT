from app.rag.embeddings import get_embedding_model
from app.rag.vectorstore import load_vector_store, add_documents_to_vectorstore
from app.rag.query import build_qa_chain, get_session_history_public
from app.rag.upload import process_uploaded_files

# --------------------------------------------------
# Embeddings
# --------------------------------------------------

print("\nLoading Embedding Model...")

embeddings = get_embedding_model()

# --------------------------------------------------
# Load Existing FAISS Index
# --------------------------------------------------

print("\nLoading FAISS Vector Store...")

vectorstore = load_vector_store(
    embeddings
)


# --------------------------------------------------
# Build QA Chain
# --------------------------------------------------

print("\nBuilding ResearchGPT QA Chain...")

qa_chain, retriever = build_qa_chain(
    vectorstore
)

print("\nResearchGPT Ready!")
print("\nAvailable Commands")

print("------------------------------")

print("papers   -> List indexed papers")

print("stats    -> Show database statistics")

print("history  -> Show conversation history")

print("clear    -> Clear memory")

print("compare  -> Compare research papers")

print("upload   -> Add new PDFs")

print("exit     -> Quit\n")


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
# Upload PDFs
# --------------------------------------------------

if question.lower() == "upload":

    folder = input(
        "\nEnter PDF folder path: "
    ).strip()

    try:

        chunks = process_uploaded_files(folder)

        vectorstore = add_documents_to_vectorstore(
            vectorstore,
            chunks
        )

        qa_chain, retriever = build_qa_chain(
            vectorstore
        )

        print("\nKnowledge Base Updated Successfully!")

    except Exception as e:

        print(f"\nError: {e}")

    continue


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

        history = get_session_history_public(
            SESSION_ID
        )

        print("\nConversation History:\n")

        if len(history.messages) == 0:

            print("No conversation history found.")

        else:

            for msg in history.messages:
                print(msg)

        continue

    # -------------------------------------------------
    # Clear Memory
    # --------------------------------------------------

    if question.lower() == "clear":

        get_session_history_public(
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
Compare the papers overall regarding:

{question}

You might provide:

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