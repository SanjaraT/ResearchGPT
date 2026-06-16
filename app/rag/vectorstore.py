from langchain_community.vectorstores import FAISS

# create vector store
def create_vector_store(chunks, embeddings):

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore

# save
def save_vector_store(vectorstore):

    vectorstore.save_local(
        "data/faiss_index"
    )

# load
def load_vector_store(embeddings):

    return FAISS.load_local(
        "data/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )