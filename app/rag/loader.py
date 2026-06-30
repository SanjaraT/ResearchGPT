from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_folder(folder):

    docs = []

    pdf_files = Path(folder).glob("*.pdf")

    for pdf_file in pdf_files:

        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))

        pages = loader.load()

        for page in pages:

            page.metadata["source_file"] = pdf_file.name

            page.metadata["folder"] = str(folder)

        docs.extend(pages)

    return docs


def load_all_pdfs():

    all_docs = []

    all_docs.extend(
        load_folder("data/pdfs")
    )

    all_docs.extend(
        load_folder("data/uploads")
    )

    return all_docs