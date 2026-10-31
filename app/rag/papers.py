"""
Paper catalogue: titles and aliases for the papers in the knowledge base.

Used for two things:
  1. Chunk headers   - every chunk is stored as "[Paper: <title>]\\n<text>", so
                       the embedding and the LLM both know which paper it is from.
  2. Query routing   - if a question names a paper (by title or alias), retrieval
                       is restricted to that paper's chunks.

Papers that are not listed here (for example user uploads) still work: they get
a header built from the file name and are never used for routing.

Add new papers by adding an entry below. Aliases are matched case-insensitively
and only as whole words, so "bert" does not match "RoBERTa" or "DistilBERT".

When one alias sits inside a longer alias of a DIFFERENT paper ("BERT" inside
"Sentence-BERT", "RAG" inside "Self-RAG", "LLaMA" inside "Llama 2"), only the
longer one counts. See detect_papers().
"""

import re

PAPERS = {
    # ------------------------------------------------------------------
    # original 14
    # ------------------------------------------------------------------
    "Attention_is_all_you_need.pdf": {
        "title": "Attention Is All You Need",
        "aliases": ["attention is all you need"],
    },
    "BERT.pdf": {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "aliases": ["bert", "pre-training of deep bidirectional transformers"],
    },
    "Few_Shot.pdf": {
        "title": "Language Models are Few-Shot Learners (GPT-3)",
        "aliases": ["language models are few-shot learners", "gpt-3", "gpt3"],
    },
    "GPT-1.pdf": {
        "title": "Improving Language Understanding by Generative Pre-Training (GPT-1)",
        "aliases": [
            "improving language understanding by generative pre-training",
            "gpt-1",
            "gpt1",
        ],
    },
    "Language_model.pdf": {
        "title": "Language Models are Unsupervised Multitask Learners (GPT-2)",
        "aliases": [
            "language models are unsupervised multitask learners",
            "gpt-2",
            "gpt2",
        ],
    },
    "LLama.pdf": {
        "title": "LLaMA: Open and Efficient Foundation Language Models",
        "aliases": ["llama", "llama 1", "llama-1", "llama1"],
    },
    "LoRA.pdf": {
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "aliases": ["lora", "low-rank adaptation"],
    },
    "RAG.pdf": {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG)",
        "aliases": [
            "retrieval-augmented generation for knowledge-intensive",
            "rag",
            "rag-sequence",
            "rag-token",
        ],
    },
    "RoBERTa.pdf": {
        "title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
        "aliases": ["roberta"],
    },
    "Transfer_Learning.pdf": {
        "title": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)",
        "aliases": [
            "exploring the limits of transfer learning",
            "text-to-text transfer transformer",
            "t5",
            "colossal clean crawled corpus",
            "c4",
        ],
    },
    "XLNet.pdf": {
        "title": "XLNet: Generalized Autoregressive Pretraining for Language Understanding",
        "aliases": ["xlnet"],
    },
    "Generative_ai.pdf": {
        "title": "Generative Adversarial Nets (GAN)",
        "aliases": [
            "generative adversarial nets",
            "generative adversarial networks",
            "gan",
            "gans",
        ],
    },
    "ImageNet.pdf": {
        "title": "ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)",
        "aliases": [
            "imagenet classification",
            "alexnet",
            "ilsvrc-2012",   # hyphenated form used by this paper; plain "ILSVRC" is also used by ResNet and YOLO
        ],
    },
    "YOLO.pdf": {
        "title": "You Only Look Once: Unified, Real-Time Object Detection (YOLO)",
        "aliases": ["you only look once", "yolo", "fast yolo"],
    },
    # ------------------------------------------------------------------
    # added 16
    # ------------------------------------------------------------------
    "ALBERT.pdf": {
        "title": "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations",
        "aliases": ["albert", "a lite bert"],
    },
    "DistilBERT.pdf": {
        "title": "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter",
        "aliases": ["distilbert", "distilled version of bert"],
    },
    "ELECTRA.pdf": {
        "title": "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators",
        "aliases": ["electra"],
    },
    "BART.pdf": {
        "title": "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension",
        "aliases": ["bart", "denoising sequence-to-sequence pre-training"],
    },
    "DPR.pdf": {
        "title": "Dense Passage Retrieval for Open-Domain Question Answering (DPR)",
        "aliases": ["dense passage retrieval", "dpr"],
    },
    "Sentence_BERT.pdf": {
        "title": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
        "aliases": [
            "sentence-bert",
            "sentence bert",
            "sbert",
            "siamese bert-networks",
        ],
    },
    "Scaling_Laws.pdf": {
        "title": "Scaling Laws for Neural Language Models",
        "aliases": ["scaling laws for neural language models", "kaplan et al"],
    },
    "Chinchilla.pdf": {
        "title": "Training Compute-Optimal Large Language Models (Chinchilla)",
        "aliases": [
            "chinchilla",
            "training compute-optimal large language models",
            "compute-optimal",
        ],
    },
    "InstructGPT.pdf": {
        "title": "Training language models to follow instructions with human feedback (InstructGPT)",
        "aliases": [
            "instructgpt",
            "training language models to follow instructions",
            "follow instructions with human feedback",
        ],
    },
    "Chain_of_Thought.pdf": {
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "aliases": ["chain-of-thought", "chain of thought"],
    },
    "Llama2.pdf": {
        "title": "Llama 2: Open Foundation and Fine-Tuned Chat Models",
        "aliases": ["llama 2", "llama-2", "llama2", "llama 2-chat"],
    },
    "QLoRA.pdf": {
        "title": "QLoRA: Efficient Finetuning of Quantized LLMs",
        "aliases": ["qlora", "efficient finetuning of quantized"],
    },
    "Self_RAG.pdf": {
        "title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        "aliases": ["self-rag", "self rag", "selfrag"],
    },
    "Lost_in_the_Middle.pdf": {
        "title": "Lost in the Middle: How Language Models Use Long Contexts",
        "aliases": ["lost in the middle"],
    },
    "ResNet.pdf": {
        "title": "Deep Residual Learning for Image Recognition (ResNet)",
        "aliases": [
            "resnet",
            "deep residual learning",
            "residual network",
            "residual networks",
        ],
    },
    "DDPM.pdf": {
        "title": "Denoising Diffusion Probabilistic Models (DDPM)",
        "aliases": [
            "ddpm",
            "denoising diffusion probabilistic models",
            "denoising diffusion",
        ],
    },
}

HEADER_PREFIX = "[Paper:"


def paper_title(source_file):
    """Human-readable title for a source file (falls back to the file name)."""

    entry = PAPERS.get(source_file)

    if entry:
        return entry["title"]

    stem = source_file.rsplit(".", 1)[0] if source_file else "Unknown"

    return stem.replace("_", " ")


def paper_header(source_file):
    """The line that is prepended to every chunk of a paper."""

    return f"{HEADER_PREFIX} {paper_title(source_file)}]"


def _alias_pattern(alias):
    # whole-word match: not glued to letters or digits on either side.
    # Hyphens and apostrophes count as boundaries ("BERT-BASE", "BERT's").
    return re.compile(
        r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


_PATTERNS = [
    (source_file, _alias_pattern(alias))
    for source_file, entry in PAPERS.items()
    for alias in entry["aliases"]
]


def detect_papers(question, available=None):
    """
    Return the source files named in the question, in order of first mention.

    available: optional set of source files that are actually indexed; names
    that are not indexed are ignored so retrieval never filters to nothing.

    Overlap rule: a match that lies inside a longer match belonging to a
    different paper is dropped. "Sentence-BERT" therefore routes to
    Sentence_BERT.pdf only, while "Sentence-BERT and BERT" routes to both.
    """

    matches = []  # (source_file, start, end)

    for source_file, pattern in _PATTERNS:

        if available is not None and source_file not in available:
            continue

        for match in pattern.finditer(question):
            matches.append((source_file, match.start(), match.end()))

    def swallowed(m):

        file, start, end = m

        return any(
            other_file != file
            and other_start <= start
            and end <= other_end
            and (other_end - other_start) > (end - start)
            for other_file, other_start, other_end in matches
        )

    first_hit = {}

    for file, start, end in matches:

        if swallowed((file, start, end)):
            continue

        if file not in first_hit or start < first_hit[file]:
            first_hit[file] = start

    return sorted(first_hit, key=first_hit.get)