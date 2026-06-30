import streamlit as st

from app.rag.upload import process_uploaded_files

from app.rag.vectorstore import (
    add_documents_to_vectorstore,
    get_existing_papers
)

from app.ui.system import reload_system


# --------------------------------------------------
# Upload Sidebar
# --------------------------------------------------

def upload_sidebar():

    st.sidebar.markdown("---")

    st.sidebar.subheader(
        "📄 Upload Research Papers"
    )

    # ------------------------------------------
    # Streamlit uploader reset key
    # ------------------------------------------

    if "uploader_key" not in st.session_state:

        st.session_state.uploader_key = 0

    uploaded_files = st.sidebar.file_uploader(

        "Choose PDF(s)",

        type=["pdf"],

        accept_multiple_files=True,

        key=f"uploader_{st.session_state.uploader_key}"

    )

    # ------------------------------------------
    # Upload Button
    # ------------------------------------------

    if st.sidebar.button(
        "➕ Add to Knowledge Base"
    ):

        if not uploaded_files:

            st.sidebar.warning(
                "Please upload at least one PDF."
            )

            return

        with st.spinner(
            "Processing papers..."
        ):

            existing = get_existing_papers(
                st.session_state.vectorstore
            )

            (
                chunks,
                added,
                skipped
            ) = process_uploaded_files(

                uploaded_files,

                existing

            )

            # ----------------------------------
            # Add chunks
            # ----------------------------------

            if len(chunks) > 0:

                add_documents_to_vectorstore(

                    st.session_state.vectorstore,

                    chunks

                )

                reload_system()

        # --------------------------------------
        # Success Messages
        # --------------------------------------

        if added:

            st.sidebar.success(

                f"Successfully indexed {len(added)} paper(s)."

            )

            for paper in added:

                st.sidebar.write(
                    f"✅ {paper}"
                )

        if skipped:

            st.sidebar.warning(

                "Already indexed:\n"

            )

            for paper in skipped:

                st.sidebar.write(
                    f"⚠ {paper}"
                )

        # --------------------------------------
        # Clear uploader
        # --------------------------------------

        st.session_state.uploader_key += 1

        st.rerun()