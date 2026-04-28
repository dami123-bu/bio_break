"""Attacker-side helpers for current PharmaHelp RAG experiments."""

from .payloads import (
    AttackPayload,
    build_retrieval_bias_payloads,
    build_stub_hijack_doc,
    build_proto_context_payloads,
)
from .stub_attack import run_stub_keyword_hijack_demo

__all__ = [
    "AttackPayload",
    "build_retrieval_bias_payloads",
    "build_stub_hijack_doc",
    "build_proto_context_payloads",
    "run_stub_keyword_hijack_demo",
]
