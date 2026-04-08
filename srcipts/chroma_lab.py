from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .payloads import (
    AttackPayload,
    CANARY_CONTEXT,
    CANARY_OVERRIDE,
    CANARY_PERSISTENCE,
    build_proto_context_payloads,
    build_retrieval_bias_payloads,
)

CANARY_MARKERS = (CANARY_OVERRIDE, CANARY_CONTEXT, CANARY_PERSISTENCE)


def _runtime() -> tuple[Any, Any, Any]:
    """Lazy runtime imports so unit tests can import this module without Chroma installed."""
    import chromadb  # type: ignore
    from chromadb.utils.embedding_functions import OllamaEmbeddingFunction  # type: ignore
    import config  # type: ignore

    return chromadb, OllamaEmbeddingFunction, config


def _client_embed_config() -> tuple[Any, Any, Any]:
    chromadb, OllamaEmbeddingFunction, config = _runtime()
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    embed_fn = OllamaEmbeddingFunction(
        url=f"{config.OLLAMA_BASE_URL}/api/embeddings",
        model_name=config.OLLAMA_EMBED_MODEL,
    )
    return client, embed_fn, config


def reset_collection(collection_name: str) -> dict[str, Any]:
    client, _, _ = _client_embed_config()
    deleted = False
    try:
        client.delete_collection(collection_name)
        deleted = True
    except Exception:
        deleted = False
    return {"collection": collection_name, "deleted": deleted}


def get_or_create_collection(collection_name: str) -> Any:
    client, embed_fn, _ = _client_embed_config()
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def get_collection(collection_name: str) -> Any:
    client, embed_fn, _ = _client_embed_config()
    return client.get_collection(name=collection_name, embedding_function=embed_fn)


def seed_lab_collection(
    source_collection_name: str,
    lab_collection_name: str,
    limit: int = 60,
    fresh: bool = False,
) -> dict[str, Any]:
    if fresh:
        reset_collection(lab_collection_name)

    source = get_collection(source_collection_name)
    lab = get_or_create_collection(lab_collection_name)

    if lab.count() > 0:
        return {
            "lab_collection": lab_collection_name,
            "seeded": False,
            "existing_count": lab.count(),
        }

    snapshot = source.get(limit=limit, include=["documents", "metadatas"])
    source_ids = snapshot.get("ids", [])
    documents = snapshot.get("documents", [])
    metadatas = snapshot.get("metadatas", [])

    if not source_ids:
        return {
            "lab_collection": lab_collection_name,
            "seeded": False,
            "existing_count": 0,
            "warning": "Source collection is empty.",
        }

    lab_ids = [f"seed::{doc_id}" for doc_id in source_ids]
    normalized_meta = []
    for meta in metadatas:
        new_meta = dict(meta or {})
        new_meta["lab_seed_source"] = source_collection_name
        normalized_meta.append(new_meta)

    lab.upsert(ids=lab_ids, documents=documents, metadatas=normalized_meta)
    return {
        "lab_collection": lab_collection_name,
        "seeded": True,
        "copied_count": len(lab_ids),
        "lab_count_after_seed": lab.count(),
    }


def inject_payloads(collection_name: str, payloads: Iterable[AttackPayload]) -> dict[str, Any]:
    collection = get_or_create_collection(collection_name)
    payloads = list(payloads)
    collection.upsert(
        ids=[payload.chroma_id for payload in payloads],
        documents=[payload.document for payload in payloads],
        metadatas=[payload.metadata | {"title": payload.title, "attack_id": payload.attack_id} for payload in payloads],
    )
    return {
        "collection": collection_name,
        "inserted_attack_ids": [payload.attack_id for payload in payloads],
        "count_after_injection": collection.count(),
    }


def query_collection(collection_name: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    collection = get_collection(collection_name)
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    hits: list[dict[str, Any]] = []
    for rank, doc_id in enumerate(ids, start=1):
        doc = documents[rank - 1] if rank - 1 < len(documents) else ""
        meta = metadatas[rank - 1] if rank - 1 < len(metadatas) else {}
        distance = distances[rank - 1] if rank - 1 < len(distances) else None
        hits.append(
            {
                "rank": rank,
                "id": doc_id,
                "distance": distance,
                "metadata": meta or {},
                "document": doc,
                "contains_canary": any(marker in (doc or "") for marker in CANARY_MARKERS),
            }
        )
    return hits


def compute_metrics(hits: list[dict[str, Any]]) -> dict[str, Any]:
    attack_hits = [
        hit
        for hit in hits
        if str(hit.get("metadata", {}).get("source", "")).startswith("attack_")
    ]
    pubmed_hits = [
        hit
        for hit in hits
        if hit.get("metadata", {}).get("source") == "pubmed"
    ]
    return {
        "attack_in_top_k": bool(attack_hits),
        "attack_top_1": bool(attack_hits) and attack_hits[0]["rank"] == 1,
        "lowest_attack_rank": min((hit["rank"] for hit in attack_hits), default=None),
        "attack_hit_count": len(attack_hits),
        "pubmed_hit_count": len(pubmed_hits),
        "retrieval_bias": round(len(attack_hits) / max(1, len(hits)), 3),
        "canary_seen": any(hit.get("contains_canary") for hit in hits),
    }


def context_preview(hits: list[dict[str, Any]], max_chars_per_doc: int = 280) -> str:
    parts = []
    for hit in hits:
        source = hit.get("metadata", {}).get("source", "unknown")
        title = hit.get("metadata", {}).get("title", "(untitled)")
        excerpt = (hit.get("document") or "")[:max_chars_per_doc].replace("\n", " ").strip()
        parts.append(f"[rank={hit['rank']} source={source} title={title}] {excerpt}")
    return "\n\n".join(parts)


def run_chroma_scenario(
    scenario: str,
    query: str,
    drug: str,
    lab_collection_name: str = "pubmed_attack_lab",
    source_collection_name: str = "pubmed",
    copy_count: int = 60,
    top_k: int = 5,
    fresh: bool = False,
    emit_context: bool = False,
) -> dict[str, Any]:
    seed_result = seed_lab_collection(
        source_collection_name=source_collection_name,
        lab_collection_name=lab_collection_name,
        limit=copy_count,
        fresh=fresh,
    )

    injected: dict[str, Any] | None = None
    if scenario == "chroma_retrieval_bias":
        injected = inject_payloads(lab_collection_name, build_retrieval_bias_payloads(drug=drug))
    elif scenario == "proto_context_poisoning":
        injected = inject_payloads(lab_collection_name, build_proto_context_payloads(drug=drug))
    elif scenario == "persistence_check":
        # Reuse whatever is already in the lab collection. No new injection.
        injected = None
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    hits = query_collection(collection_name=lab_collection_name, query=query, top_k=top_k)
    metrics = compute_metrics(hits)

    result: dict[str, Any] = {
        "scenario": scenario,
        "query": query,
        "drug": drug,
        "lab_collection": lab_collection_name,
        "seed": seed_result,
        "injection": injected,
        "metrics": metrics,
        "hits": hits,
    }
    if emit_context:
        result["context_preview"] = context_preview(hits)
    return result


def dump_json_report(report: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path
