import os
from operator import itemgetter

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.chat_message_histories import (
    ChatMessageHistory,
)

from app.rag.config import (
    GROQ_MODEL,
)

from app.rag.reranker import rerank_documents

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
# Prompt
# --------------------------------------------------

SYSTEM_PROMPT = """
You are ResearchGPT, an AI research assistant.

Answer ONLY using the retrieved research papers.

Rules:

1. Never use outside knowledge.

2. If the answer is unavailable, reply exactly:

"I could not find the answer in the provided papers."

3. Answer in clear academic language.

4. Preserve important technical names,
datasets,
metrics,
architectures,
model names and findings.

5. Use previous conversation whenever the
user asks follow-up questions.

6. Resolve references such as:

- it
- they
- that paper
- this model
- those results

using the conversation history.

7. If multiple papers discuss the topic,
mention each paper separately.

Context:

{context}
"""


# --------------------------------------------------
# Build QA Chain
# --------------------------------------------------

def build_qa_chain(vectorstore):

    # -----------------------------
    # LLM
    # -----------------------------

    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )

    # -----------------------------
    # Retriever
    # -----------------------------

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "fetch_k": 35,
        },
    )

    # -----------------------------
    # Prompt
    # -----------------------------

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT,
            ),

            MessagesPlaceholder(
                variable_name="history"
            ),

            (
                "human",
                "{question}",
            ),
        ]
    )

    # -----------------------------
    # Format Documents
    # -----------------------------

    def format_docs(docs):

        return "\n\n".join(
            doc.page_content
            for doc in docs
        )

    # -----------------------------
    # Retrieval + Re-ranking
    # -----------------------------

    def retrieve(inputs):

        question = inputs["question"]

        docs = retriever.invoke(
            question
        )

        docs = rerank_documents(
            question,
            docs,
            top_k=5,
        )

        return format_docs(
            docs
        )

    # -----------------------------
    # Core Chain
    # -----------------------------

    rag_chain = (

        {
            "context": RunnableLambda(
                retrieve
            ),

            "question": itemgetter(
                "question"
            ),

            "history": itemgetter(
                "history"
            ),
        }

        | prompt

        | llm

        | StrOutputParser()

    )

    # -----------------------------
    # Memory
    # -----------------------------

    chain = RunnableWithMessageHistory(

        rag_chain,

        get_session_history,

        input_messages_key="question",

        history_messages_key="history",

    )

    return chain, retriever


# --------------------------------------------------
# Public Helper
# --------------------------------------------------

def get_session_history_public(
    session_id: str,
):

    return get_session_history(
        session_id
    )


# from operator import itemgetter
# import os

# from dotenv import load_dotenv
# from langchain_groq import ChatGroq

# from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_core.runnables import RunnableLambda


# from langchain_community.chat_message_histories import ChatMessageHistory

# from app.rag.config import (
#     LLM_PROVIDER,
#     GROQ_MODEL
# )
# from app.rag.reranker import rerank_documents

# load_dotenv()

# # --------------------------------------------------
# # Session Store
# # --------------------------------------------------

# store = {}


# def get_session_history(session_id: str):

#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()

#     return store[session_id]


# # --------------------------------------------------
# # Build QA Chain
# # --------------------------------------------------
# def build_qa_chain(vectorstore):

#     # ---------------------------------------
#     # Groq LLM
#     # ---------------------------------------

#     llm = ChatGroq(
#         model=GROQ_MODEL,
#         api_key=os.getenv("GROQ_API_KEY"),
#         temperature=0
#     )

#     # ---------------------------------------
#     # Retriever
#     # ---------------------------------------

#     retriever = vectorstore.as_retriever(
#         search_type="mmr",
#         search_kwargs={
#             "k": 12,
#             "fetch_k": 40
#         }
#     )

#     # ---------------------------------------
#     # Prompt
#     # ---------------------------------------

#     prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """
# You are an expert research assistant.

# Use ONLY the provided context from the retrieved research papers.

# Rules:

# 1. Answer strictly from the provided context.
# 2. Do not use outside knowledge.
# 3. Be factual and accurate.
# 4. Do not hallucinate or invent information.
# 5. If the answer is not found in the paper, reply exactly:

# "I could not find the answer in the provided paper."
# 6. Explain the answer in 2-5 sentences when sufficient information is available.
# 7. Preserve important technical terms, model names, datasets, objectives, and findings from the papers.
# 8. When comparing concepts, clearly mention the differences and improvements.
# 9. Use conversation history when the user asks follow-up questions.
# 10. Resolve references such as "it", "they", "that paper", "the model", "those results" using conversation history.
# 11. If a follow-up question refers to a previous paper or concept, continue the discussion naturally.

# Context:
# {context}
# """
#         ),

#             MessagesPlaceholder(
#                 variable_name="history"
#             ),

#             (
#                 "human",
#                 "{question}"
#             )
#         ]
#     )

#     # ---------------------------------------
#     # Format Documents
#     # ---------------------------------------

#     def format_docs(docs):

#         return "\n\n".join(
#             doc.page_content
#             for doc in docs
#         )

#     # ---------------------------------------
#     # Core RAG Chain
#     # ---------------------------------------

#     def retrieve_and_rerank(inputs):
#         question = inputs["question"]

#         history = inputs.get(
#             "history",
#             []
#         )

#         history_text = "\n".join(
#             [
#                 msg.content
#                 for msg in history[-4:]
#             ]
#         )

#         search_query = f"""
#     Conversation:
#     {history_text}

#     Current Question:
#     {question}
#     """
#         docs = retriever.invoke(
#             search_query
#         )

#         docs = rerank_documents(
#             question,
#             docs,
#             top_k=5
#         )

#         return format_docs(docs)

#     rag_chain = (

#         {
#             "context":
#                 RunnableLambda(
#                     retrieve_and_rerank
#                 ),

#             "question":
#                 itemgetter("question"),
            
#             "history":
#                 itemgetter("history")
               
#         }

#         | prompt
#         | llm
#         | StrOutputParser()

#     )

#     # ---------------------------------------
#     # Memory Wrapper
#     # ---------------------------------------

#     chain = RunnableWithMessageHistory(

#         rag_chain,

#         get_session_history,

#         input_messages_key="question",

#         history_messages_key="history"
#     )

#     return chain, retriever


# # --------------------------------------------------
# # Public Helper
# # --------------------------------------------------

# def get_session_history_public(
#     session_id: str
# ):
#     return get_session_history(session_id)