"""Heartbeat sender — llama.cpp edge variant."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from mining_types import (
    AttestationFreshness,
    Capability,
    LoadSnapshot,
    OffChainHeartbeat,
    Quantization,
    WatchdogState,
)

from infer_worker_llamacpp.config import WorkerConfig


def _quantization_for(name: str) -> Quantization:
    return {
        "FP16": Quantization.FP16,
        "FP8": Quantization.FP8,
        "INT8": Quantization.INT8,
        "INT4": Quantization.INT4,
    }.get(name, Quantization.INT4)


def build_heartbeat(config: WorkerConfig, load: LoadSnapshot) -> OffChainHeartbeat:
    now_ms = int(time.time() * 1000)
    hb = OffChainHeartbeat(
        operator_id=config.operator_id,
        capabilities=[
            Capability(
                base_model_id=config.model_id,
                quantization=_quantization_for(config.quantization),
                max_context_tokens=4096,
                max_concurrent_requests=2,
                deterministic_mode=config.deterministic_mode,
            )
        ],
        current_load=load,
        kv_cache_pressure=0.0,  # edge has no GPU memory pressure
        attestation_freshness=AttestationFreshness(
            last_attested_at_ms=now_ms,
            expires_at_ms=now_ms + 7 * 86400 * 1000,
            current_report_hash=config.attestation_report_hash,
        ),
        watchdog_state=WatchdogState(vllm_pid_alive=True, vllm_last_log_ms=now_ms),
        endpoint_url=config.base_url,
        price_per_million_tokens=config.price_per_million_tokens,
        geo_region="US",
    )
    return hb.sign(config.operator_private_key_hex)


class HeartbeatPusher:
    def __init__(
        self,
        config: WorkerConfig,
        gateway_url: str,
        load_provider: Any,
        interval_s: float | None = None,
    ) -> None:
        self.config = config
        self.gateway_url = gateway_url
        self.load_provider = load_provider
        self.interval_s = interval_s or config.heartbeat_interval_s
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self.last_hb: OffChainHeartbeat | None = None

    async def _loop(self) -> None:
        async with httpx.AsyncClient(timeout=2.0) as client:
            while not self._stop.is_set():
                hb = build_heartbeat(self.config, self.load_provider())
                self.last_hb = hb
                try:
                    await client.post(
                        f"{self.gateway_url}/internal/heartbeat",
                        json=hb.model_dump(mode="json"),
                    )
                except httpx.HTTPError:
                    pass
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.interval_s)
                except TimeoutError:
                    pass

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except TimeoutError:
                self._task.cancel()
