from operator import itemgetter
import os

from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory


from langchain_community.chat_message_histories import ChatMessageHistory

from app.rag.config import (
    LLM_PROVIDER,
    OLLAMA_MODEL,
    GROQ_MODEL
)

load_dotenv()

# --------------------------------------------------
# Session Store
# --------------------------------------------------

store = {}


def get_session_history(session_id: str):

    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    return store[session_id]


# --------------------------------------------------
# Build QA Chain
# --------------------------------------------------

def build_qa_chain(vectorstore):

    # ---------------------------------------
    # LLM Selection
    # ---------------------------------------

    if LLM_PROVIDER == "ollama":

        llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0
        )

    elif LLM_PROVIDER == "groq":

        llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )

    else:

        raise ValueError(
            f"Unsupported provider: {LLM_PROVIDER}"
        )

    # ---------------------------------------
    # Retriever
    # ---------------------------------------

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 30
        }
    )

    # ---------------------------------------
    # Prompt
    # ---------------------------------------

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert research assistant.

Use ONLY the provided context.

Rules:
1. Be factual.
2. Be concise.
3. Do not hallucinate.
4. If the answer is not found, say:

"I could not find the answer in the provided papers."

Context:
{context}
"""
            ),

            MessagesPlaceholder(
                variable_name="history"
            ),

            (
                "human",
                "{question}"
            )
        ]
    )

    # ---------------------------------------
    # Format Documents
    # ---------------------------------------

    def format_docs(docs):

        return "\n\n".join(
            doc.page_content
            for doc in docs
        )

    # ---------------------------------------
    # Core RAG Chain
    # ---------------------------------------

    rag_chain = (

        {
            "context":
                itemgetter("question")
                | retriever
                | format_docs,

            "question":
                itemgetter("question"),

            "history":
                itemgetter("history")
        }

        | prompt
        | llm
        | StrOutputParser()

    )

    # ---------------------------------------
    # Memory Wrapper
    # ---------------------------------------

    chain = RunnableWithMessageHistory(

        rag_chain,

        get_session_history,

        input_messages_key="question",

        history_messages_key="history"
    )

    return chain, retriever


# --------------------------------------------------
# Public Helper
# --------------------------------------------------

def get_session_history_public(
    session_id: str
):
    return get_session_history(session_id)