import streamlit as st


def show_stats(vectorstore):

    total_chunks = len(
        vectorstore.docstore._dict
    )

    total_docs = len(

        set(

            doc.metadata.get(
                "source_file",
                "Unknown"
            )

            for doc in vectorstore.docstore._dict.values()

        )

    )

    st.sidebar.divider()

    st.sidebar.subheader(
        "Statistics"
    )

    st.sidebar.write(
        f"📄 Papers : {total_docs}"
    )

    st.sidebar.write(
        f"🧩 Chunks : {total_chunks}"
    )