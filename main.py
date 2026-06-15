from app.rag.loader import load_pdf
from app.rag.splitter import split_documents
from app.rag.embeddings import get_embedding_model
from app.rag.vectorstore import create_vector_store
from app.rag.query import build_qa_chain

PDF_PATH = "D:\A\DL\ResearchGPT\data\pdfs\paper_1.pdf"

# print("Loading PDF...")
docs = load_pdf(PDF_PATH)

# print("Document loaded")

# print("Splitting...")
chunks = split_documents(docs)

print(f"Total Chunks: {len(chunks)}")

print("Loading Embeddings...")
embeddings = get_embedding_model()

print("Creating FAISS...")
vectorstore = create_vector_store(
    chunks,
    embeddings
)

print("Building QA Chain...")
qa_chain = build_qa_chain(
    vectorstore
)

print("\nResearchGPT Ready!")

while True:

    question = input("\nAsk Question: ")

    if question.lower() == "exit":
        break

    response = qa_chain.invoke(question)

    print("\nAnswer:")
    print(response["result"])