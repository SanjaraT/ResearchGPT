import math
import re
from typing import Any, Callable, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.rag.papers import HEADER_PREFIX, detect_papers, paper_header
from app.rag.vectorstore import get_existing_papers

_SUMMARY_RE = re.compile(r"\b(summari[sz]e|summary|overview)\b", re.IGNORECASE)

# number of opening chunks (abstract / introduction) always included for a
# "summarize <paper>" question; the text of "summarize X" matches nothing useful
SUMMARY_LEAD_CHUNKS = 2


def default_rerank(question, docs, top_k):
    # imported lazily: loading the cross-encoder is slow and needs a model download
    from app.rag.reranker import rerank_documents

    return rerank_documents(question, docs, top_k=top_k)


class PaperAwareRetriever(BaseRetriever):
    """
    Retrieval that knows which papers a question is about.

    * Question names one or more indexed papers (by title or alias):
      search only inside each named paper, rerank within it, and keep the best
      chunks of every named paper, so comparisons see both sides.
    * Question names no paper: the original MMR search + cross-encoder rerank.

    The same object is used by the QA chain and by the UI, so the sources shown
    to the user are the chunks the model actually saw.
    """

    vectorstore: Any
    top_k: int = 5
    k: int = 10                 # candidates per search
    mmr_fetch_k: int = 35       # MMR pool when no paper is named
    filter_fetch_k: int = 400   # FAISS filters AFTER the nearest-neighbour fetch
    rerank_fn: Optional[Callable] = None

    # ------------------------------------------------------------------

    def _rerank(self, question, docs, top_k):

        fn = self.rerank_fn or default_rerank

        return fn(question, docs, top_k)

    def routed_papers(self, question):

        return detect_papers(
            question,
            available=get_existing_papers(self.vectorstore),
        )

    # ------------------------------------------------------------------

    def retrieve(self, question, top_k=None) -> List[Document]:

        top_k = top_k or self.top_k

        papers = self.routed_papers(question)

        if not papers:

            docs = self.vectorstore.max_marginal_relevance_search(
                question,
                k=self.k,
                fetch_k=self.mmr_fetch_k,
            )

            return self._rerank(question, docs, top_k)

        per_paper = max(2, math.ceil(top_k / len(papers)))

        results = []

        for source_file in papers:

            docs = self.vectorstore.similarity_search(
                question,
                k=max(self.k, per_paper),
                fetch_k=self.filter_fetch_k,
                filter={"source_file": source_file},
            )

            picked = self._rerank(question, docs, per_paper)

            if len(papers) == 1 and _SUMMARY_RE.search(question):

                picked = self._with_opening_chunks(
                    source_file,
                    picked,
                    top_k,
                )

            results.extend(picked)

        return results

    # ------------------------------------------------------------------

    def _with_opening_chunks(self, source_file, picked, top_k):
        """First chunks of the paper (abstract) + best remaining chunks."""

        lead = []

        for doc in self.vectorstore.docstore._dict.values():

            if doc.metadata.get("source_file") == source_file:

                lead.append(doc)

                if len(lead) == SUMMARY_LEAD_CHUNKS:
                    break

        seen = {d.page_content for d in lead}

        rest = [d for d in picked if d.page_content not in seen]

        return (lead + rest)[:top_k]

    # ------------------------------------------------------------------

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:

        return self.retrieve(query)


def format_docs(docs):
    """Context string for the LLM: every excerpt is labelled with its paper."""

    parts = []

    for doc in docs:

        text = doc.page_content

        if not text.startswith(HEADER_PREFIX):

            text = (
                paper_header(doc.metadata.get("source_file"))
                + "\n"
                + text
            )

        parts.append(text)

    return "\n\n".join(parts)
