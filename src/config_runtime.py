from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class RuntimeConfig:
    pharmahelp_root: Path | None
    chroma_dir: Path | None
    ollama_base_url: str
    ollama_embed_model: str
    source_collection: str
    lab_collection: str

def _expand(path: str | None) -> Path | None:
    if not path: return None
    return Path(path).expanduser().resolve()

def discover_pharmahelp_root() -> Path | None:
    candidates = [_expand(os.getenv("PHARMAHELP_ROOT")), Path.cwd().parent / "pharma_help", Path.home() / "ec521" / "pharma_help"]
    for c in candidates:
        if c and c.exists(): return c.resolve()
    return None

def discover_chroma_dir(root: Path | None) -> Path | None:
    env_dir = _expand(os.getenv("PHARMAHELP_CHROMA_DIR"))
    candidates = [env_dir, root / "data" / "chroma" if root else None, Path.cwd() / "data" / "chroma", Path.cwd().parent / "pharma_help" / "data" / "chroma", Path.home() / "ec521" / "pharma_help" / "data" / "chroma"]
    for c in candidates:
        if c and c.exists(): return c.resolve()
    return env_dir or (root / "data" / "chroma" if root else None)

def load_runtime_config() -> RuntimeConfig:
    root = discover_pharmahelp_root()
    return RuntimeConfig(
        pharmahelp_root=root,
        chroma_dir=discover_chroma_dir(root),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        source_collection=os.getenv("PHARMAATTACK_SOURCE_COLLECTION", "pubmed"),
        lab_collection=os.getenv("PHARMAATTACK_LAB_COLLECTION", "pubmed_attack_lab"),
    )
