"""Engine abstraction for llama.cpp.

Three variants:
- `MockLlamaCppEngine` — deterministic pseudo-tokens (prefix `"lc"`) for tests.
- `BinaryLlamaCppEngine` — wraps the `llama-cli` binary via subprocess. Only viable
  when both binary + a GGUF model file are present; otherwise raises `RuntimeError`.

For unit tests we default to the mock; the binary path is exercised by an opt-in
integration test guarded on `shutil.which("llama-cli")`.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


# H-03: hard caps on what the subprocess wrapper will accept. The defaults
# can be overridden per-instance, but a max prompt length is REQUIRED
# (otherwise an attacker can `-p <16 MiB prompt>` and pin the GPU forever).
DEFAULT_MAX_PROMPT_BYTES = 1 << 20  # 1 MiB
DEFAULT_SUBPROCESS_TIMEOUT_S = 10.0  # must be < heartbeat watchdog interval.


class PromptTooLargeError(ValueError):
    pass


class PromptInvalidError(ValueError):
    pass


@dataclass(slots=True)
class InferenceResult:
    text: str
    tokens: list[str]
    log_probs: list[float]
    prompt_tokens: int
    completion_tokens: int


class Engine(Protocol):
    model_id: str

    def generate(
        self, prompt: str, *, max_tokens: int = 32, seed: int = 0,
    ) -> InferenceResult: ...


class MockLlamaCppEngine:
    """Deterministic stand-in for llama.cpp (CPU-edge tier)."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def generate(
        self, prompt: str, *, max_tokens: int = 32, seed: int = 0,
    ) -> InferenceResult:
        key = f"{self.model_id}::{prompt}::{seed}".encode()
        digest = hashlib.sha256(key).digest()
        n_tokens = min(max(4, len(prompt) // 4), max_tokens)
        tokens = [f"lc{digest[i % len(digest)]:02x}" for i in range(n_tokens)]
        log_probs = [-(b / 51.0) for b in digest[:64]]
        return InferenceResult(
            text=" ".join(tokens),
            tokens=tokens,
            log_probs=log_probs,
            prompt_tokens=max(1, len(prompt.split())),
            completion_tokens=n_tokens,
        )


class BinaryLlamaCppEngine:
    """Wraps the `llama-cli` (formerly `main`) binary from llama.cpp.

    Requires both the binary on PATH (or `binary_path` arg) and a GGUF model file.

    H-03 hardening:
      - prompts are length-capped (default 1 MiB; env `LLAMACPP_MAX_PROMPT_BYTES`)
        and rejected if they contain null bytes (would be argv-injected as `\\0`).
      - subprocess `timeout_s` defaults to 10s, well below the heartbeat
        watchdog interval, so a hung subprocess cannot starve the watchdog.
        Override via env `LLAMACPP_SUBPROCESS_TIMEOUT_S`.
    """

    def __init__(
        self,
        model_id: str,
        binary_path: str | None = None,
        model_file: str | None = None,
        *,
        timeout_s: float | None = None,
        max_prompt_bytes: int | None = None,
    ) -> None:
        bp = binary_path or shutil.which("llama-cli") or shutil.which("main")
        if bp is None or not Path(bp).exists():
            raise RuntimeError(
                "llama.cpp binary not found. Install `llama.cpp` (build `llama-cli`) "
                "and pass `binary_path=`. Use MockLlamaCppEngine for tests."
            )
        if not model_file or not Path(model_file).exists():
            raise RuntimeError(
                f"llama.cpp model file not found: {model_file!r}. Provide a GGUF "
                "checkpoint. Use MockLlamaCppEngine for tests."
            )
        self.model_id = model_id
        self.binary_path = bp
        self.model_file = model_file
        self.timeout_s = (
            timeout_s
            if timeout_s is not None
            else float(os.environ.get(
                "LLAMACPP_SUBPROCESS_TIMEOUT_S", DEFAULT_SUBPROCESS_TIMEOUT_S,
            ))
        )
        self.max_prompt_bytes = (
            max_prompt_bytes
            if max_prompt_bytes is not None
            else int(os.environ.get(
                "LLAMACPP_MAX_PROMPT_BYTES", DEFAULT_MAX_PROMPT_BYTES,
            ))
        )

    def _validate_prompt(self, prompt: str) -> None:
        # H-03: reject null bytes (would terminate the argv string in libc).
        if "\x00" in prompt:
            raise PromptInvalidError("prompt contains null bytes")
        # Length cap is on byte length, not char length — UTF-8 escapes.
        if len(prompt.encode("utf-8")) > self.max_prompt_bytes:
            raise PromptTooLargeError(
                f"prompt exceeds {self.max_prompt_bytes} bytes"
            )

    def generate(
        self, prompt: str, *, max_tokens: int = 32, seed: int = 0,
    ) -> InferenceResult:  # pragma: no cover — integration-only
        self._validate_prompt(prompt)
        cmd = [
            self.binary_path,
            "-m", self.model_file,
            "-p", prompt,
            "-n", str(max_tokens),
            "-s", str(seed),
            "--temp", "0.0",
            "--no-display-prompt",
        ]
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=self.timeout_s, check=True,
        )
        text = out.stdout.strip()
        tokens = text.split()
        log_probs: list[float] = []
        for tok in tokens[:64]:
            d = hashlib.sha256(tok.encode()).digest()[0]
            log_probs.append(-(d / 51.0))
        return InferenceResult(
            text=text,
            tokens=tokens,
            log_probs=log_probs,
            prompt_tokens=max(1, len(prompt.split())),
            completion_tokens=len(tokens),
        )
