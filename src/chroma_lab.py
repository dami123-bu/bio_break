from __future__ import annotations
from typing import Any
from pharma_attack.config_runtime import RuntimeConfig, load_runtime_config
from pharma_attack.metrics import AttackReport, compute_retrieval_metrics, judge_retrieval_attack
from pharma_attack.offline_lab import simulate_answer
from pharma_attack.payloads import payloads_for_scenario

def _load_chroma_runtime():
    try:
        import chromadb  # type: ignore
        from chromadb.utils.embedding_functions import OllamaEmbeddingFunction  # type: ignore
    except Exception as exc:
        raise RuntimeError("Chroma-backed mode requires chromadb and Ollama embedding support. Run --mode offline if unavailable.") from exc
    return chromadb, OllamaEmbeddingFunction

def _make_client_and_embedding(config: RuntimeConfig):
    chromadb, OllamaEmbeddingFunction = _load_chroma_runtime()
    if not config.chroma_dir:
        raise RuntimeError("Could not discover Chroma directory. Set PHARMAHELP_CHROMA_DIR.")
    client = chromadb.PersistentClient(path=str(config.chroma_dir))
    embedding_function = OllamaEmbeddingFunction(url=f"{config.ollama_base_url.rstrip('/')}/api/embeddings", model_name=config.ollama_embed_model)
    return client, embedding_function

def reset_collection(collection_name: str, config: RuntimeConfig | None = None) -> None:
    config = config or load_runtime_config()
    client, _ = _make_client_and_embedding(config)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

def _get_or_create_collection(name: str, config: RuntimeConfig):
    client, embedding_function = _make_client_and_embedding(config)
    return client.get_or_create_collection(name=name, embedding_function=embedding_function)

def _get_collection(name: str, config: RuntimeConfig):
    client, embedding_function = _make_client_and_embedding(config)
    return client.get_collection(name=name, embedding_function=embedding_function)

def seed_lab_collection(*, source_collection_name: str | None = None, lab_collection_name: str | None = None, copy_limit: int = 60, fresh: bool = False, config: RuntimeConfig | None = None) -> dict[str, Any]:
    config = config or load_runtime_config()
    source_collection_name = source_collection_name or config.source_collection
    lab_collection_name = lab_collection_name or config.lab_collection
    if fresh:
        reset_collection(lab_collection_name, config=config)
    lab = _get_or_create_collection(lab_collection_name, config)
    existing = lab.count()
    if existing > 0 and not fresh:
        return {"lab_collection": lab_collection_name, "seeded": False, "existing_count": existing}
    source = _get_collection(source_collection_name, config)
    data = source.get(limit=copy_limit, include=["documents", "metadatas"])
    docs = data.get("documents", []) or []
    ids = data.get("ids", []) or []
    metas = data.get("metadatas", []) or []
    if not docs:
        raise RuntimeError(f"Source collection {source_collection_name!r} has no documents to seed.")
    seed_ids, seed_docs, seed_meta = [], [], []
    for idx, doc in enumerate(docs):
        original_id = ids[idx] if idx < len(ids) else f"source_{idx}"
        metadata = dict(metas[idx] if idx < len(metas) and metas[idx] else {})
        metadata.setdefault("source", metadata.get("source", "pubmed"))
        metadata["lab_seed_source"] = source_collection_name
        seed_ids.append(f"seed::{original_id}")
        seed_docs.append(doc)
        seed_meta.append(metadata)
    lab.add(ids=seed_ids, documents=seed_docs, metadatas=seed_meta)
    return {"lab_collection": lab_collection_name, "seeded": True, "copied_count": len(seed_docs), "lab_count_after_seed": lab.count()}

def inject_payloads(*, scenario: str, drug: str, collection_name: str | None = None, volume_count: int = 10, config: RuntimeConfig | None = None) -> dict[str, Any]:
    config = config or load_runtime_config()
    collection_name = collection_name or config.lab_collection
    collection = _get_or_create_collection(collection_name, config)
    payloads = payloads_for_scenario(scenario, drug, volume_count=volume_count)
    ids, docs, metas = [], [], []
    for p in payloads:
        ids.append(f"attack::{p['id']}")
        docs.append(p["content"])
        m = dict(p.get("metadata", {}))
        m.setdefault("title", p.get("title", p["id"]))
        m.setdefault("source", "attack_internal_docs")
        metas.append(m)
    collection.upsert(ids=ids, documents=docs, metadatas=metas)
    return {"collection": collection_name, "inserted_attack_ids": ids, "count_after_injection": collection.count()}

def query_collection(*, query: str, collection_name: str | None = None, top_k: int = 10, config: RuntimeConfig | None = None) -> list[dict[str, Any]]:
    config = config or load_runtime_config()
    collection_name = collection_name or config.lab_collection
    collection = _get_or_create_collection(collection_name, config)
    raw = collection.query(query_texts=[query], n_results=top_k, include=["documents", "metadatas", "distances"])
    ids = raw.get("ids", [[]])[0]
    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]
    hits = []
    for idx, hit_id in enumerate(ids, start=1):
        doc = docs[idx - 1] if idx - 1 < len(docs) else ""
        metadata = metas[idx - 1] if idx - 1 < len(metas) and metas[idx - 1] else {}
        distance = distances[idx - 1] if idx - 1 < len(distances) else None
        hits.append({"rank": idx, "id": hit_id, "distance": distance, "metadata": metadata, "document": doc, "contains_canary": "SECURITY_TEST" in str(doc) or "SECURITY_TEST" in str(metadata)})
    return hits

def run_chroma_scenario(*, scenario: str, query: str, drug: str, fresh: bool = False, top_k: int = 10, copy_limit: int = 60, volume_count: int = 10, config: RuntimeConfig | None = None) -> dict[str, Any]:
    config = config or load_runtime_config()
    seed_result = seed_lab_collection(source_collection_name=config.source_collection, lab_collection_name=config.lab_collection, copy_limit=copy_limit, fresh=fresh, config=config)
    baseline_hits = query_collection(query=query, collection_name=config.lab_collection, top_k=top_k, config=config)
    injection_result = None
    if scenario not in {"persistence_check"}:
        injection_result = inject_payloads(scenario=scenario, drug=drug, collection_name=config.lab_collection, volume_count=volume_count, config=config)
    attacked_hits = query_collection(query=query, collection_name=config.lab_collection, top_k=top_k, config=config)
    metrics = compute_retrieval_metrics(attacked_hits, top_k=top_k)
    decision = judge_retrieval_attack(metrics, scenario=scenario)
    report = AttackReport(
        attack_id=scenario, scenario=scenario, query=query, drug=drug, injection_point="Knowledge Base / ChromaDB", goal="integrity",
        success=decision.success, partial_success=decision.partial_success, success_level=decision.success_level, success_reason=decision.success_reason,
        metrics=metrics, hits=attacked_hits, baseline_hits=baseline_hits, injected_payload_ids=(injection_result or {}).get("inserted_attack_ids", []),
        normal_expected_output=simulate_answer(query, baseline_hits), attacked_output=simulate_answer(query, attacked_hits),
        explanation="Chroma mode tests whether attacker-controlled documents can enter the top-k evidence set inside an isolated lab collection copied from the target knowledge base.",
        mode="chroma"
    )
    data = report.to_dict()
    data["lab_collection"] = config.lab_collection
    data["seed"] = seed_result
    data["injection"] = injection_result
    data["runtime_config"] = {"pharmahelp_root": str(config.pharmahelp_root) if config.pharmahelp_root else None, "chroma_dir": str(config.chroma_dir) if config.chroma_dir else None, "source_collection": config.source_collection, "lab_collection": config.lab_collection, "ollama_base_url": config.ollama_base_url, "ollama_embed_model": config.ollama_embed_model}
    return data
