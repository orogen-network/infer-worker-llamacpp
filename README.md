# infer-worker-llamacpp

llama.cpp backed operator daemon for the **edge tier** (CPU-only / Apple Silicon / small
embedded GPU). Specializes in quantized (INT4 / Q4_K_M) chat + small embedding workloads.

## Architecture

Mirrors `infer-worker-vllm` (FastAPI + signed receipts + heartbeat) but with two
edge-specific engine implementations:

- **`MockLlamaCppEngine`** — deterministic (prefix `"lc"`); used by tests + dev boxes.
- **`BinaryLlamaCppEngine`** — wraps the `llama-cli` (formerly `main`) binary via
  subprocess. Requires both binary on PATH and a GGUF model file; raises `RuntimeError`
  otherwise.

We default to the binary-wrapping path (rather than `llama-cpp-python` Python bindings)
because the binary is cheap to ship in containers and isolates llama.cpp's build matrix.

## Edge tier semantics

- Declared `quantization = INT4`, `max_concurrent_requests = 2`, `max_context_tokens = 4096`.
- Receipt's `gpu_model` field set to `"cpu-edge"` so validators can apply tier-specific
  fault thresholds (edge log-prob drift tolerance is laxer than dc-premium per RFC-0008).

## Endpoints

Same as `infer-worker-vllm`: `/v1/chat/completions`, `/healthz`, `/internal/last_heartbeat`.

## Bring-your-own-binary

```python
engine = BinaryLlamaCppEngine(
    model_id="llama-3-8b-q4",
    binary_path="/usr/local/bin/llama-cli",
    model_file="/var/lib/edge/models/llama-3-8b-q4.gguf",
)
```
