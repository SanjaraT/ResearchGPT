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

