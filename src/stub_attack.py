from __future__ import annotations
from typing import Any
from pharma_attack.payloads import build_stub_keyword_doc

BASELINE_DOCS: list[dict[str, Any]] = [
    {"id": "doc1", "title": "Aspirin and gastrointestinal effects", "content": "Aspirin may cause stomach upset, nausea, and gastrointestinal irritation."},
    {"id": "doc2", "title": "Ibuprofen precautions", "content": "Ibuprofen may increase risk of stomach bleeding in some patients."},
    {"id": "doc3", "title": "Tylenol uses", "content": "Tylenol is commonly used to reduce pain and fever."},
]
STOP_WORDS = {"the", "is", "a", "an", "and", "or", "of", "to", "for", "what", "how", "about", "today", "again", "are"}

def tokenize(text: str) -> list[str]:
    return [w.strip("?.!,;:()[]{}\"'").lower() for w in text.split() if w.strip("?.!,;:()[]{}\"'").lower() not in STOP_WORDS]

def retrieve_docs_lexical(query: str, docs: list[dict[str, Any]], k: int = 3) -> list[dict[str, Any]]:
    query_words = tokenize(query)
    scored: list[tuple[int, dict[str, Any]]] = []
    for doc in docs:
        text = f"{doc.get('title', '')} {doc.get('content', '')}".lower()
        score = sum(1 for word in query_words if word and word in text)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:k]]

def synthesize_answer(query: str, docs: list[dict[str, Any]]) -> str:
    if not docs:
        return "I could not find relevant documents."
    top = docs[0]
    return f"Based on {top['title']}, the answer to '{query}' is: {top['content']}"

def _hits(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"rank": i, "id": d.get("id"), "metadata": d.get("metadata", {}), "title": d.get("title"), "document": d.get("content"), "contains_canary": "SECURITY_TEST" in str(d.get("content", ""))} for i, d in enumerate(docs, 1)]

def run_stub_keyword_hijack_demo(query: str, drug: str, k: int = 2) -> dict[str, Any]:
    baseline_docs = retrieve_docs_lexical(query, BASELINE_DOCS, k=k)
    baseline_answer = synthesize_answer(query, baseline_docs)
    attack_doc = build_stub_keyword_doc(drug)
    attacked_corpus = BASELINE_DOCS + [{"id": attack_doc["id"], "title": attack_doc["title"], "content": attack_doc["content"], "metadata": attack_doc["metadata"]}]
    attacked_docs = retrieve_docs_lexical(query, attacked_corpus, k=k)
    attacked_answer = synthesize_answer(query, attacked_docs)
    success = "SECURITY_TEST" in attacked_answer
    return {
        "scenario": "stub_keyword_hijack", "query": query, "drug": drug, "mode": "stub",
        "malicious_input": attack_doc,
        "normal_expected_output": baseline_answer,
        "attacked_output": attacked_answer,
        "baseline": {"retrieved_docs": baseline_docs, "hits": _hits(baseline_docs), "answer": baseline_answer, "canary_in_answer": "SECURITY_TEST" in baseline_answer},
        "attacked": {"retrieved_docs": attacked_docs, "hits": _hits(attacked_docs), "answer": attacked_answer, "canary_in_answer": success},
        "attack_success": success, "success": success, "partial_success": False,
        "success_level": "strong" if success else "failure",
        "explanation": "The payload is keyword-dense and becomes the top lexical result; the answer generator copies the top document."
    }
