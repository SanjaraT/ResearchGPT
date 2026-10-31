LLM_PROVIDER = "groq"
GROQ_MODEL = "openai/gpt-oss-20b"

# Model used to grade answers in evaluation/evaluate.py.
JUDGE_MODEL = "openai/gpt-oss-120b"

LLM_KWARGS = (
    {"reasoning_effort": "low", "max_tokens": 2048}
    if "gpt-oss" in GROQ_MODEL
    else {}
)

# Used when the first answer comes back empty (retry once with more room).
RETRY_MAX_TOKENS = 4096

# Retrieval
TOP_K = 5          # chunks sent to the LLM
WIDE_TOP_K = 8     # chunks on the single retry after a refusal
MAX_HISTORY_MESSAGES = 6
