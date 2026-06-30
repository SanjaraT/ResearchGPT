import os
import shutil

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from app.rag.splitter import split_documents

UPLOAD_FOLDER = Path("data/uploads")

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

# --------------------------------------------------
# Process Uploaded PDFs
# --------------------------------------------------

def process_uploaded_files(
    uploaded_files,
    existing_papers,
):
    """
    Processes uploaded PDFs.

    Returns
    -------
    chunks
        List of LangChain Document chunks.

    added
        Newly accepted paper names.

    skipped
        Duplicate paper names.
    """

    all_documents = []

    added = []

    skipped = []

    for uploaded_file in uploaded_files:

        filename = uploaded_file.name

        # --------------------------------------
        # Duplicate Check
        # --------------------------------------

        if filename in existing_papers:

            skipped.append(filename)

            continue

        # ----------------------------------
        # Save permanently
        # ----------------------------------

        save_path = UPLOAD_FOLDER / filename

        with open(save_path, "wb") as f:

            f.write(uploaded_file.getbuffer())

        # --------------------------------------
        # Load PDF
        # --------------------------------------

        loader = PyPDFLoader(
            save_path
        )

        documents = loader.load()

        # --------------------------------------
        # Add metadata
        # --------------------------------------

        for doc in documents:

            doc.metadata["source_file"] = filename

        all_documents.extend(
            documents
        )

        added.append(
            filename
        )


    # ------------------------------------------
    # Split into chunks
    # ------------------------------------------

    if len(all_documents) == 0:

        return [], added, skipped

    chunks = split_documents(
        all_documents
    )

    return (
        chunks,
        added,
        skipped,
    )