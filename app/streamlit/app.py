import streamlit as st
from rag.query import build_qa_chain

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="ResearchGPT",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 ResearchGPT - AI Research Assistant")
st.caption("Ask questions from your knowledge base using RAG + LLM")

# -------------------------------
# SESSION STATE
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = build_qa_chain()

# -------------------------------
# SIDEBAR SETTINGS
# -------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    model_name = st.selectbox(
        "LLM Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
    )

    temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.2)

    st.markdown("---")
    st.info("Upload PDFs or update vector DB in backend scripts")

# -------------------------------
# DISPLAY CHAT HISTORY
# -------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------
# USER INPUT
# -------------------------------
user_query = st.chat_input("Ask anything from your research data...")

if user_query:
    # store user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    # ---------------------------
    # GET ANSWER FROM RAG CHAIN
    # ---------------------------
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤔"):

            result = st.session_state.qa_chain({"query": user_query})

            answer = result["result"]

            st.markdown(answer)

    # store assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )