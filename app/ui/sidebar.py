import streamlit as st

from app.rag.query import (
    get_session_history_public
)


def sidebar(session_id):

    st.sidebar.title(
        "ResearchGPT"
    )

    if st.sidebar.button(
        "Clear Memory"
    ):

        get_session_history_public(
            session_id
        ).clear()

        st.session_state.messages = []

        st.sidebar.success(
            "Memory Cleared"
        )
        