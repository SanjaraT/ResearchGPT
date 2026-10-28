import json
import re
from pathlib import Path

PAPERS = {
    r"attention is all you need|\btransformer\b": "Attention_is_all_you_need.pdf",
    r"\bbert\b": "BERT.pdf",
    r"\broberta\b": "RoBERTa.pdf",
    r"\bxlnet\b": "XLNet.pdf",
    r"\bt5\b": "Transfer_Learning.pdf",
    r"\brag\b|retrieval-augmented": "RAG.pdf",
    r"\bllama\b": "LLama.pdf",
    r"\blora\b": "LoRA.pdf",
    r"\bgpt\b": "GPT-1.pdf",
}

path = Path(__file__).resolve().parent / "evaluation_set.json"
data = json.loads(path.read_text(encoding="utf-8"))

unmatched = []
for item in data:
    q = item["question"].lower()
    files = sorted({f for pattern, f in PAPERS.items() if re.search(pattern, q)})
    item["source_files"] = files
    if not files:
        unmatched.append(item["question"])

path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Labelled {len(data) - len(unmatched)}/{len(data)} questions.")
for q in unmatched:
    print("  NEEDS MANUAL LABEL:", q)