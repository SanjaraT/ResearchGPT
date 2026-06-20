from app.rag.loader import load_all_pdfs
from app.rag.splitter import split_documents
from app.rag.embeddings import get_embedding_model
from app.rag.vectorstore import (
    create_vector_store,
    save_vector_store
)

PDF_PATH = "data/pdfs"

print("Loading PDFs...")
docs = load_all_pdfs(PDF_PATH)

print("Splitting...")
chunks = split_documents(docs)

print("Loading embeddings...")
embeddings = get_embedding_model()

print("Creating FAISS...")
vectorstore = create_vector_store(
    chunks,
    embeddings
)

save_vector_store(vectorstore)

print("Vectorstore saved!")