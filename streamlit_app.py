import uuid
import streamlit as st

from app.ui.system import initialize_system
from app.ui.sidebar import sidebar
from app.ui.knowledge_base import show_papers
from app.ui.stats import show_stats
from app.ui.upload import upload_sidebar
from app.ui.chat import chat_interface


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(

    page_title="ResearchGPT",

    page_icon="📚",

    layout="wide"

)

st.title("📚 ResearchGPT")

st.caption(
    "Research Assistant powered by RAG + Groq"
)

# --------------------------------------------------
# Session ID
# --------------------------------------------------

if "session_id" not in st.session_state:

    st.session_state.session_id = str(
        uuid.uuid4()
    )

# --------------------------------------------------
# Initialize System
# --------------------------------------------------

initialize_system()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

sidebar(
    st.session_state.session_id
)

# --------------------------------------------------
# Knowledge Base
# --------------------------------------------------

show_papers(
    st.session_state.vectorstore
)

show_stats(
    st.session_state.vectorstore
)

# --------------------------------------------------
# Upload Papers
# --------------------------------------------------

upload_sidebar()

# --------------------------------------------------
# Chat
# --------------------------------------------------

chat_interface()

# import streamlit as st
# from app.rag.embeddings import get_embedding_model
# from app.rag.vectorstore import load_vector_store
# from app.rag.query import build_qa_chain, get_session_history_public
# import uuid


# # ----------------------------------
# # Page
# # ----------------------------------

# st.set_page_config(
#     page_title="ResearchGPT",
#     page_icon="📚",
#     layout="wide"
# )

# st.title("📚 ResearchGPT")
# st.caption(
#     "Research Assistant powered by RAG + Groq"
# )

# # ----------------------------------
# # Load Resources
# # ----------------------------------

# @st.cache_resource
# def load_system():

#     embeddings = get_embedding_model()

#     vectorstore = load_vector_store(
#         embeddings
#     )

#     qa_chain, retriever = build_qa_chain(
#         vectorstore
#     )

#     return qa_chain, retriever, vectorstore


# qa_chain, retriever, vectorstore = load_system()

# # ----------------------------------
# # Sidebar
# # ----------------------------------

# st.sidebar.title("ResearchGPT")

# if st.sidebar.button("Clear Memory"):


#     get_session_history_public(
#         st.session_state.session_id
#     ).clear()

#     st.session_state.messages = []

#     st.sidebar.success(
#         "Memory Cleared"
#     )

# # ----------------------------------
# # Show Papers
# # ----------------------------------

# all_sources = set()

# for doc in vectorstore.docstore._dict.values():

#     source = doc.metadata.get(
#         "source_file",
#         "Unknown"
#     )

#     all_sources.add(source)

# st.sidebar.subheader("Indexed Papers")

# for paper in sorted(all_sources):

#     st.sidebar.write(
#         f"📄 {paper}"
#     )

# # ----------------------------------
# # Chat History
# # ----------------------------------

# if "messages" not in st.session_state:

#     st.session_state.messages = []

# if "session_id" not in st.session_state:

#     st.session_state.session_id = str(
#         uuid.uuid4()
#     )

# for msg in st.session_state.messages:

#     with st.chat_message(
#         msg["role"]
#     ):
#         st.markdown(
#             msg["content"]
#         )

# # ----------------------------------
# # User Question
# # ----------------------------------

# question = st.chat_input(
#     "Ask a research question..."
# )

# if question:

#     st.session_state.messages.append(
#         {
#             "role": "user",
#             "content": question
#         }
#     )

#     with st.chat_message("user"):

#         st.markdown(question)

#     with st.spinner(
#         "ResearchGPT is thinking..."
#     ):

#         docs = retriever.invoke(
#             question
#         )

#         answer = qa_chain.invoke(
#             {
#                 "question": question
#             },
#             config={
#                 "configurable": {
#                     "session_id":
#                     st.session_state.session_id
#                 }
#             }
#         )

#     with st.chat_message(
#         "assistant"
#     ):

#         st.markdown(answer)

#         st.markdown("---")

#         st.markdown(
#             "**Sources**"
#         )

#         seen = set()

#         for doc in docs:

#             source = doc.metadata.get(
#                 "source_file",
#                 "Unknown"
#             )

#             page = (
#                 doc.metadata.get(
#                     "page",
#                     0
#                 ) + 1
#             )

#             citation = (
#                 f"{source} "
#                 f"(Page {page})"
#             )

#             if citation not in seen:

#                 st.write(
#                     f"• {citation}"
#                 )

#                 seen.add(citation)

#     st.session_state.messages.append(
#         {
#             "role": "assistant",
#             "content": answer
#         }
#     )