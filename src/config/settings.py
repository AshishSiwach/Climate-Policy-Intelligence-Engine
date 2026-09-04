"""
Centralised settings for CPIE — Phase 0 Baseline Hardening.

Single source of truth for all configuration values. Loaded from
``configs/config.yaml``; default values match the locked Week 2 decisions.

Usage::

    from config.settings import get_settings
    s = get_settings()
    print(s.synthesis.max_tokens)   # 2000
    print(s.settings_fingerprint()) # e.g. "a3f1c2b4"

The ``settings_fingerprint()`` returns the first 8 hex chars of the SHA-256
of the JSON-serialised settings dict.  Include it in every log record so you
can tell which config version produced a query.
"""

from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Default YAML path — relative to project root.
_DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


# ---------------------------------------------------------------------------
# Sub-models — one per top-level key in config.yaml
# ---------------------------------------------------------------------------


class ChunkingSettings(BaseModel):
    default_chunk_size: int = 400
    default_overlap: int = 80
    max_chunk_size: int = 512
    min_chunk_size: int = 50


class EmbeddingSettings(BaseModel):
    model: str = "BAAI/bge-base-en-v1.5"
    device: str = "cuda"


class BM25Settings(BaseModel):
    k1: float = 1.5
    b: float = 0.75


class RRFSettings(BaseModel):
    k: int = 60


class RerankerSettings(BaseModel):
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_candidates: int = 20
    top_k_final: int = 5
    lazy_load: bool = True


class RetrievalSettings(BaseModel):
    bm25: BM25Settings = Field(default_factory=BM25Settings)
    rrf: RRFSettings = Field(default_factory=RRFSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)


class VectorStoreSettings(BaseModel):
    provider: str = "chroma"
    persist_directory: str = "data/processed/chroma_db"
    collection_name: str = "cpie"


class SynthesisSettings(BaseModel):
    model: str = "gpt-5.4-mini"
    # CANONICAL VALUE: 2000 tokens.
    # config.yaml had max_tokens: 512, which was OBSOLETE — the runtime always
    # used 2000 to avoid LengthFinishReasonError on structured output (bumped
    # from 800 in Week 5 Step 3b). config.yaml's 512 is now commented out.
    max_tokens: int = 2000
    temperature: float = 0.0


class OutputSettings(BaseModel):
    include_contradictions: bool = True


class MonitoringSettings(BaseModel):
    log_dir: str = "logs/"
    log_format: str = "jsonl"
    dashboard_port: int = 8502


class UISettings(BaseModel):
    port: int = 8501
    title: str = "CPIE — Climate Policy Intelligence Engine"


# ---------------------------------------------------------------------------
# Top-level Settings model
# ---------------------------------------------------------------------------


class Settings(BaseModel):
    """Centralised settings model. Load via :func:`get_settings`."""

    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    synthesis: SynthesisSettings = Field(default_factory=SynthesisSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    ui: UISettings = Field(default_factory=UISettings)

    @classmethod
    def from_yaml(cls, path: str | Path = _DEFAULT_CONFIG_PATH) -> "Settings":
        """Load settings from a YAML file. Missing file → use all defaults."""
        path = Path(path)
        if not path.exists():
            logger.warning("Config file not found at %s — using defaults", path)
            return cls()
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        # Discard keys not in the model (forward-compat: old keys won't crash)
        known_keys = set(cls.model_fields)
        data = {k: v for k, v in raw.items() if k in known_keys}
        return cls(**data)

    def settings_fingerprint(self) -> str:
        """Return the first 8 hex chars of SHA-256(JSON(settings)).

        Tag every log record with this to identify which config version
        produced a query — useful when config changes mid-experiment.
        """
        serialised = json.dumps(self.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(serialised.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_settings(config_path: str = str(_DEFAULT_CONFIG_PATH)) -> Settings:
    """Return the cached Settings singleton. Thread-safe (lru_cache is GIL-held).

    Call ``get_settings.cache_clear()`` in tests to reset between runs.
    """
    return Settings.from_yaml(config_path)


def settings_fingerprint(config_path: str = str(_DEFAULT_CONFIG_PATH)) -> str:
    """Convenience wrapper: fingerprint of the cached settings."""
    return get_settings(config_path).settings_fingerprint()
