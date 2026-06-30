import streamlit as st


def show_papers(vectorstore):

    papers = sorted(

        set(

            doc.metadata.get(
                "source_file",
                "Unknown"
            )

            for doc in vectorstore.docstore._dict.values()

        )

    )

    st.sidebar.subheader(
        "Knowledge Base"
    )

    for paper in papers:

        st.sidebar.write(
            f"📄 {paper}"
        )