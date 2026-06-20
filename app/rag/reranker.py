from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

def rerank_documents(question, docs, top_k=5):

    pairs = [
        (question, doc.page_content)
        for doc in docs
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, docs),
        reverse=True,
        key=lambda x: x[0]
    )

    return [
        doc
        for score, doc in ranked[:top_k]
    ]