"""Offline E5/FAISS search adapter for Champion core."""
from __future__ import annotations
import json
import urllib.request
from typing import Sequence
RETRIEVER_URL = "http://127.0.0.1:8766"
TOP_K = 5
SNIPPET_CHARS = 400
def batch_search(queries: Sequence[str], engines=None) -> str:
    """Execute each upstream-proposed query in original order and return evidence text."""
    blocks = []
    for query in queries:
        body = json.dumps({"query": query, "top_n": TOP_K, "return_score": True}).encode()
        request = urllib.request.Request(RETRIEVER_URL + "/search", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=90) as response:
            docs, _scores = json.load(response)
        lines = [f"Search results for: {query}"]
        for rank, doc in enumerate(docs, 1):
            title, _, content = str(doc.get("contents", "")).partition("\n")
            snippet = content[:SNIPPET_CHARS].replace("\n", " ").strip()
            lines.append(f"{rank}. {title}\n{snippet}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
