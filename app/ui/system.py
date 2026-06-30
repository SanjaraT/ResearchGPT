import streamlit as st

from app.rag.embeddings import get_embedding_model
from app.rag.vectorstore import load_vector_store
from app.rag.query import build_qa_chain


# --------------------------------------------------
# Cached Embedding Model
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():

    return get_embedding_model()


# --------------------------------------------------
# Initialize ResearchGPT
# --------------------------------------------------

def initialize_system():
    """
    Loads the embedding model, vectorstore,
    retriever and QA chain.

    This only runs once per Streamlit session.
    """

    if "vectorstore" in st.session_state:
        return

    embeddings = load_embedding_model()

    vectorstore = load_vector_store(
        embeddings
    )

    qa_chain, retriever = build_qa_chain(
        vectorstore
    )

    st.session_state.embeddings = embeddings

    st.session_state.vectorstore = vectorstore

    st.session_state.qa_chain = qa_chain

    st.session_state.retriever = retriever


# --------------------------------------------------
# Reload System
# --------------------------------------------------

def reload_system():
    """
    Rebuilds the vectorstore, retriever
    and QA chain after new papers
    have been uploaded.
    """

    embeddings = st.session_state.embeddings

    vectorstore = load_vector_store(
        embeddings
    )

    qa_chain, retriever = build_qa_chain(
        vectorstore
    )

    st.session_state.vectorstore = vectorstore

    st.session_state.qa_chain = qa_chain

    st.session_state.retriever = retriever