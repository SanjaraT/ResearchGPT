from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.papers import HEADER_PREFIX, paper_header


def add_paper_headers(chunks):
    """
    Prepend "[Paper: <title>]" to every chunk (once).

    Body text of a paper rarely contains its own title, so without this the
    embedding and the LLM cannot tell which paper a chunk came from.
    """

    for chunk in chunks:

        if chunk.page_content.startswith(HEADER_PREFIX):
            continue

        header = paper_header(
            chunk.metadata.get("source_file")
        )

        chunk.page_content = f"{header}\n{chunk.page_content}"

    return chunks


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    return add_paper_headers(chunks)
