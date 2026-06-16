from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def build_qa_chain(vectorstore):

    llm = ChatOllama(
        model="phi3",
        temperature=0
    )

    prompt_template = """
You are an expert research assistant.

Use ONLY the provided context to answer the question.

Rules:
1. If the answer exists in the context, provide the exact information.
2. Be concise and factual.
3. Do not make up information.
4. If the answer is not present, respond:
   "I could not find the answer in the provided papers."

Context:
{context}

Question:
{question}

Answer:
"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20
        }
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": PROMPT
        }
    )

    return qa_chain