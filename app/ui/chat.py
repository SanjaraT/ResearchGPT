import streamlit as st


# --------------------------------------------------
# Chat Interface
# --------------------------------------------------

def chat_interface():

    # ------------------------------------------
    # Initialize Chat History
    # ------------------------------------------

    if "messages" not in st.session_state:

        st.session_state.messages = []

    # ------------------------------------------
    # Display Previous Messages
    # ------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # ------------------------------------------
    # User Input
    # ------------------------------------------

    question = st.chat_input(
        "Ask a research question..."
    )

    if not question:

        return

    # ------------------------------------------
    # Show User Message
    # ------------------------------------------

    st.session_state.messages.append(

        {
            "role": "user",
            "content": question
        }

    )

    with st.chat_message("user"):

        st.markdown(question)

    # ------------------------------------------
    # Get Latest Objects
    # ------------------------------------------

    qa_chain = st.session_state.qa_chain

    retriever = st.session_state.retriever

    # ------------------------------------------
    # Retrieve Documents
    # ------------------------------------------

    with st.spinner(
        "ResearchGPT is thinking..."
    ):

        docs = retriever.invoke(
            question
        )

        answer = qa_chain.invoke(

            {
                "question": question
            },

            config={
                "configurable": {

                    "session_id":
                    st.session_state.session_id

                }
            }

        )

    # ------------------------------------------
    # Assistant Response
    # ------------------------------------------

    with st.chat_message("assistant"):

        st.markdown(answer)

        if docs:

            st.markdown("---")

            st.markdown(
                "#### 📚 Sources"
            )

            seen = set()

            for doc in docs:

                source = doc.metadata.get(
                    "source_file",
                    "Unknown"
                )

                page = (
                    doc.metadata.get(
                        "page",
                        0
                    ) + 1
                )

                citation = (
                    f"**{source}** "
                    f"(Page {page})"
                )

                if citation not in seen:

                    st.markdown(
                        f"- {citation}"
                    )

                    seen.add(
                        citation
                    )

    # ------------------------------------------
    # Save Assistant Message
    # ------------------------------------------

    st.session_state.messages.append(

        {
            "role": "assistant",
            "content": answer
        }

    )