from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA


def build_qa_chain(vectorstore):

    llm = ChatOllama(
        model="phi3"
    )

    retriever = vectorstore.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    return qa_chain