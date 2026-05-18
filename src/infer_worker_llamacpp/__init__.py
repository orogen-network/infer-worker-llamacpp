"""llama.cpp backed operator daemon for edge / embed-only."""

from infer_worker_llamacpp.app import build_app
from infer_worker_llamacpp.config import WorkerConfig
from infer_worker_llamacpp.engine import (
    BinaryLlamaCppEngine,
    Engine,
    InferenceResult,
    MockLlamaCppEngine,
)

__version__ = "0.1.0"

__all__ = [
    "BinaryLlamaCppEngine",
    "Engine",
    "InferenceResult",
    "MockLlamaCppEngine",
    "WorkerConfig",
    "build_app",
]
