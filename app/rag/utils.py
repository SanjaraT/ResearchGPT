def inspect_chunks(chunks, n=3):

    for i, chunk in enumerate(chunks[:n]):

        print("\n" + "="*50)

        print(f"Chunk {i+1}")

        print(chunk.metadata)

        print(chunk.page_content[:300])