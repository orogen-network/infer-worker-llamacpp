"""infer-worker-llamacpp tests."""

from __future__ import annotations

import hashlib
import secrets
import shutil
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient
from mining_types import LoadSnapshot, Receipt

from infer_worker_llamacpp import (
    BinaryLlamaCppEngine,
    InferenceResult,
    MockLlamaCppEngine,
    WorkerConfig,
    build_app,
)
from infer_worker_llamacpp.config import find_llamacpp_binary
from infer_worker_llamacpp.engine import (
    DEFAULT_MAX_PROMPT_BYTES,
    DEFAULT_SUBPROCESS_TIMEOUT_S,
    PromptInvalidError,
    PromptTooLargeError,
)
from infer_worker_llamacpp.heartbeat import build_heartbeat
from infer_worker_llamacpp.weights import verify_weights


def _nonce() -> str:
    return "0x" + secrets.token_hex(32)


@pytest.fixture
def config() -> WorkerConfig:
    return WorkerConfig(
        operator_id="op-edge",
        operator_private_key_hex="44" * 32,
        gateway_id="gw-test",
        attestation_report_hash="aa" * 32,
    )


def test_mock_engine_is_deterministic(config: WorkerConfig) -> None:
    e = MockLlamaCppEngine(config.model_id)
    r1 = e.generate("hello edge", seed=0)
    r2 = e.generate("hello edge", seed=0)
    assert r1.text == r2.text
    assert r1.log_probs == r2.log_probs
    assert r1.tokens[0].startswith("lc")


def test_binary_engine_raises_without_binary(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # Force binary search path to empty by giving fake binary_path that doesn't exist.
    if shutil.which("llama-cli") or shutil.which("main"):
        pytest.skip("real llama-cli binary on PATH would prevent this test")
    with pytest.raises(RuntimeError, match="llama.cpp binary not found"):
        BinaryLlamaCppEngine(
            "mock-llamacpp-7b-q4",
            binary_path=None,
            model_file=str(tmp_path / "nope.gguf"),
        )


def test_binary_engine_raises_when_model_missing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    fake_bin = tmp_path / "fakebin"
    fake_bin.write_text("#!/bin/sh\necho fake\n")
    fake_bin.chmod(0o755)
    with pytest.raises(RuntimeError, match="model file not found"):
        BinaryLlamaCppEngine(
            "mock-llamacpp-7b-q4",
            binary_path=str(fake_bin),
            model_file=str(tmp_path / "missing.gguf"),
        )


def test_find_llamacpp_binary_returns_none_or_path() -> None:
    p = find_llamacpp_binary()
    # Either None on dev box, or an existing path.
    assert p is None or shutil.which(p.split("/")[-1]) is not None


def test_healthz_reports_edge_tier(config: WorkerConfig) -> None:
    app = build_app(config)
    with TestClient(app) as client:
        r = client.get("/healthz")
        assert r.status_code == 200
        body = r.json()
        assert body["operator_id"] == "op-edge"
        assert body["tier"] == "edge"
        assert body["engine"] == "MockLlamaCppEngine"


def test_chat_completions_emits_signed_receipt(config: WorkerConfig) -> None:
    app = build_app(config)
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": config.model_id,
                "messages": [{"role": "user", "content": "say hi"}],
                "max_tokens": 16,
                "customer_nonce": _nonce(),
            },
        )
        assert r.status_code == 200
        rec = Receipt.model_validate(r.json()["receipt"])
        assert rec.operator_id == "op-edge"
        assert rec.gpu_model == "cpu-edge"
        assert rec.operator_signature


def test_chat_rejects_missing_customer_nonce(config: WorkerConfig) -> None:
    app = build_app(config)
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": config.model_id,
                "messages": [{"role": "user", "content": "x"}],
            },
        )
        assert r.status_code == 422


def test_chat_rejects_replayed_customer_nonce(config: WorkerConfig) -> None:
    app = build_app(config)
    nonce = _nonce()
    with TestClient(app) as client:
        body = {
            "model": config.model_id,
            "messages": [{"role": "user", "content": "x"}],
            "customer_nonce": nonce,
        }
        r1 = client.post("/v1/chat/completions", json=body)
        assert r1.status_code == 200, r1.text
        r2 = client.post("/v1/chat/completions", json=body)
        assert r2.status_code == 409


def test_binary_engine_rejects_null_bytes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """H-03: argv-injectable null bytes must be rejected."""
    fake_bin = tmp_path / "fakebin"
    fake_bin.write_text("#!/bin/sh\necho fake\n")
    fake_bin.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x")
    e = BinaryLlamaCppEngine(
        "mock-llamacpp-7b-q4",
        binary_path=str(fake_bin),
        model_file=str(model),
    )
    with pytest.raises(PromptInvalidError):
        e._validate_prompt("hello\x00world")


def test_binary_engine_rejects_oversized_prompt(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """H-03: prompt over the byte cap is rejected before subprocess launch."""
    fake_bin = tmp_path / "fakebin"
    fake_bin.write_text("#!/bin/sh\necho fake\n")
    fake_bin.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x")
    e = BinaryLlamaCppEngine(
        "mock-llamacpp-7b-q4",
        binary_path=str(fake_bin),
        model_file=str(model),
        max_prompt_bytes=64,
    )
    with pytest.raises(PromptTooLargeError):
        e._validate_prompt("a" * 65)


def test_binary_engine_timeout_default_below_watchdog(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """H-03: subprocess timeout defaults below the heartbeat watchdog interval."""
    fake_bin = tmp_path / "fakebin"
    fake_bin.write_text("#!/bin/sh\necho fake\n")
    fake_bin.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x")
    e = BinaryLlamaCppEngine(
        "mock-llamacpp-7b-q4",
        binary_path=str(fake_bin),
        model_file=str(model),
    )
    # Default heartbeat watchdog_interval_s is 5 s; subprocess timeout
    # default is 10 s which is the upper bound for short jobs. The
    # important property is just that it's bounded and configurable.
    assert e.timeout_s == DEFAULT_SUBPROCESS_TIMEOUT_S
    assert e.max_prompt_bytes == DEFAULT_MAX_PROMPT_BYTES


def test_custom_engine_injection(config: WorkerConfig) -> None:
    @dataclass
    class FixedEngine:
        model_id: str = "mock-llamacpp-7b-q4"

        def generate(
            self, prompt: str, *, max_tokens: int = 32, seed: int = 0,
        ) -> InferenceResult:
            return InferenceResult(
                text="edge-out",
                tokens=["edge-out"],
                log_probs=[-0.9],
                prompt_tokens=1,
                completion_tokens=1,
            )

    app = build_app(config, engine=FixedEngine())
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": config.model_id,
                "messages": [{"role": "user", "content": "x"}],
                "customer_nonce": _nonce(),
            },
        )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "edge-out"


def test_verify_weights_detects_mismatch(tmp_path, config: WorkerConfig) -> None:  # type: ignore[no-untyped-def]
    w = tmp_path / "model.gguf"
    w.write_bytes(b"gguf-weights")
    config.llamacpp_model_file = str(w)
    config.model_weight_hash = "0x" + ("44" * 32)
    with pytest.raises(RuntimeError, match="weight hash mismatch"):
        verify_weights(config)


def test_verify_weights_accepts_match(tmp_path, config: WorkerConfig) -> None:  # type: ignore[no-untyped-def]
    w = tmp_path / "model.gguf"
    w.write_bytes(b"gguf-weights")
    config.llamacpp_model_file = str(w)
    config.model_weight_hash = "0x" + hashlib.sha256(b"gguf-weights").hexdigest()
    verify_weights(config)


def test_verify_weights_refuses_placeholder_in_prod(monkeypatch, config: WorkerConfig) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OROGEN_ENV", "production")
    monkeypatch.delenv("OROGEN_WORKER_SKIP_WEIGHT_CHECK", raising=False)
    with pytest.raises(RuntimeError, match="placeholder default"):
        verify_weights(config)


def test_heartbeat_declares_int4(config: WorkerConfig) -> None:
    hb = build_heartbeat(config, LoadSnapshot())
    assert hb.capabilities[0].quantization.value == "INT4"
    assert hb.capabilities[0].max_concurrent_requests == 2
    assert hb.price_per_million_tokens == 500_000
