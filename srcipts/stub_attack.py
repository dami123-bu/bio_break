from __future__ import annotations

from copy import deepcopy
from typing import Any

from .payloads import CANARY_OVERRIDE, build_stub_hijack_doc

BASELINE_DOCS: list[dict[str, str]] = [
    {
        "id": "doc1",
        "title": "Aspirin and gastrointestinal effects",
        "content": "Aspirin may cause stomach upset, nausea, and gastrointestinal irritation.",
    },
    {
        "id": "doc2",
        "title": "Ibuprofen precautions",
        "content": "Ibuprofen may increase risk of stomach bleeding in some patients.",
    },
    {
        "id": "doc3",
        "title": "Tylenol uses",
        "content": "Tylenol is commonly used to reduce pain and fever.",
    },
]

STOP_WORDS = {
    "the",
    "is",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "for",
    "what",
    "how",
    "about",
    "today",
    "again",
    "are",
    "again?",
    "hows",
}


def tokenize_query(query: str) -> list[str]:
    return [
        word.strip("?.!,").lower()
        for word in query.split()
        if word.strip("?.!,").lower() not in STOP_WORDS
    ]


def retrieve_docs_with_docs(query: str, docs: list[dict[str, str]], k: int = 2) -> list[dict[str, str]]:
    query_words = tokenize_query(query)
    scored_matches: list[tuple[int, dict[str, str]]] = []
    for doc in docs:
        text = f"{doc['title']} {doc['content']}".lower()
        score = sum(1 for word in query_words if word in text)
        if score > 0:
            scored_matches.append((score, doc))
    scored_matches.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored_matches[:k]]


def synthesize_from_topdoc(query: str, docs: list[dict[str, str]], history: list[dict[str, str]]) -> str:
    if not docs:
        return "I could not find relevant documents."
    top_doc = docs[0]
    if len(history) > 1:
        return (
            f"Considering the conversation so far, based on {top_doc['title']}, "
            f"the answer to '{query}' is: {top_doc['content']}"
        )
    return f"Based on {top_doc['title']}, the answer to '{query}' is: {top_doc['content']}"


def _snapshot(docs: list[dict[str, str]], query: str) -> dict[str, Any]:
    hits = retrieve_docs_with_docs(query=query, docs=docs, k=2)
    answer = synthesize_from_topdoc(query=query, docs=hits, history=[{"role": "user", "content": query}])
    top_doc = hits[0] if hits else None
    return {
        "retrieved_docs": hits,
        "top_doc": top_doc,
        "answer": answer,
        "canary_in_answer": CANARY_OVERRIDE in answer,
    }


def run_stub_keyword_hijack_demo(query: str, drug: str) -> dict[str, Any]:
    baseline_docs = deepcopy(BASELINE_DOCS)
    poisoned_docs = deepcopy(BASELINE_DOCS)
    poisoned_docs.append(build_stub_hijack_doc(drug=drug))

    baseline = _snapshot(baseline_docs, query=query)
    attacked = _snapshot(poisoned_docs, query=query)

    attack_success = bool(attacked["top_doc"]) and attacked["top_doc"]["id"].startswith("attack_stub_")
    return {
        "scenario": "stub_keyword_hijack",
        "query": query,
        "drug": drug,
        "baseline": baseline,
        "attacked": attacked,
        "attack_success": attack_success,
    }
