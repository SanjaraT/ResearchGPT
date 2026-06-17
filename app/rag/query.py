from langchain_ollama import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory.buffer import ConversationBufferMemory


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
            "k": 8,
            "fetch_k": 30
        }
    )


    memory =  ConversationBufferMemory(
        memory_key = "chat_history",
        return_messages = True,
        output_key = "answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = retriever,
        memory = memory,
        return_source_documents = True,
        combine_docs_chain_kwargs ={
            "prompt": PROMPT
        }
    )

    return chain