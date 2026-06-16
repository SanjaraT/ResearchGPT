from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

def load_all_pdfs(pdf_folder):

    all_docs = []
    pdf_files = Path(pdf_folder).glob("*.pdf")

    for  pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))

        docs = loader.load()


        # source information
        for doc in docs:

            doc.metadata["source_file"] = pdf_file.name
        all_docs.extend(docs)

    return all_docs




# def load_pdf(pdf_path):
#     loader = PyPDFLoader(pdf_path)
#     docs = loader.load()
#     return docs