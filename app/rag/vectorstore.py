from pathlib import Path

from langchain_community.vectorstores import FAISS


# --------------------------------------------------
# Constants
# --------------------------------------------------

INDEX_PATH = Path("data/faiss_index")


# --------------------------------------------------
# Create New Vector Store
# --------------------------------------------------

def create_vector_store(chunks, embeddings):
    """
    Creates a brand-new FAISS index.
    Used only when creating the index for the first time.
    """

    return FAISS.from_documents(
        chunks,
        embeddings
    )


# --------------------------------------------------
# Save
# --------------------------------------------------

def save_vector_store(vectorstore):
    """
    Saves the FAISS index locally.
    """

    INDEX_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    vectorstore.save_local(
        str(INDEX_PATH)
    )


# --------------------------------------------------
# Load
# --------------------------------------------------

def load_vector_store(embeddings):
    """
    Loads an existing FAISS index.
    """

    return FAISS.load_local(
        str(INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )


# --------------------------------------------------
# Add Documents
# --------------------------------------------------

def add_documents_to_vectorstore(
    vectorstore,
    chunks
):
    """
    Incrementally adds new chunks to the
    existing FAISS index.
    """

    if not chunks:
        return vectorstore

    vectorstore.add_documents(
        chunks
    )

    save_vector_store(
        vectorstore
    )

    return vectorstore


# --------------------------------------------------
# Existing Papers
# --------------------------------------------------

def get_existing_papers(
    vectorstore
):
    """
    Returns the filenames of every indexed paper.
    Used to prevent duplicate uploads.
    """

    papers = set()

    for doc in vectorstore.docstore._dict.values():

        source = doc.metadata.get(
            "source_file"
        )

        if source:
            papers.add(source)

    return papers


# --------------------------------------------------
# Statistics
# --------------------------------------------------

def get_vectorstore_statistics(
    vectorstore
):
    """
    Returns useful statistics about the knowledge base.
    """

    papers = get_existing_papers(
        vectorstore
    )

    return {

        "papers": len(papers),

        "chunks": len(
            vectorstore.docstore._dict
        )

    }


