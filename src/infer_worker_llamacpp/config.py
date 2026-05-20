"""Worker configuration — llama.cpp (edge tier)."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field


@dataclass(slots=True)
class WorkerConfig:
    operator_id: str
    operator_private_key_hex: str
    gateway_id: str
    attestation_report_hash: str
    model_id: str = "mock-llamacpp-7b-q4"
    model_weight_hash: str = "0x" + "ab" * 32
    kernel_pack_hash: str = "0x" + "cd" * 32
    heartbeat_interval_s: float = 12.0
    base_url: str = ""
    capabilities: list[str] = field(default_factory=lambda: ["mock-llamacpp-7b-q4"])
    gateway_auth_token: str = ""
    deterministic_mode: bool = True
    # llama.cpp typically runs INT4 quantized on edge.
    quantization: str = "INT4"
    price_per_million_tokens: int = 500_000
    # If set, BinaryLlamaCppEngine will try to invoke this binary.
    llamacpp_binary_path: str | None = None
    llamacpp_model_file: str | None = None


def find_llamacpp_binary() -> str | None:
    """Look for `llama-cli` (newer) or `main` (older) on PATH."""
    for name in ("llama-cli", "llama", "main"):
        p = shutil.which(name)
        if p:
            return p
    return None
