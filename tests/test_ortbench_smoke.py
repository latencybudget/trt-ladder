"""End-to-end smoke test for the ONNX Runtime comparison path.

Unlike test_parse_and_report.py (which tests parsing/report logic against
captured trtexec text), this exercises real code against a real model: build a
tiny synthetic ONNX graph, load it in onnxruntime, and time it. There is no
Jetson or TensorRT involved, so this is CPU-only and skips cleanly wherever
onnxruntime isn't installed -- but where it runs, the numbers are real.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
MODEL_PATH = os.path.join(EXAMPLES_DIR, "toy_mlp.onnx")


def _ensure_model():
    if not os.path.isfile(MODEL_PATH):
        subprocess.run(
            [sys.executable, os.path.join(EXAMPLES_DIR, "make_toy_model.py")],
            check=True,
        )


def test_ortbench_runs_on_real_model_or_skips_cleanly():
    from trt_ladder import ortbench

    if not ortbench.HAVE_ORT:
        print("[SKIP] onnxruntime not installed")
        return

    _ensure_model()
    providers = ortbench.available_providers()
    provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in providers else "CPUExecutionProvider"

    result = ortbench.bench(MODEL_PATH, provider=provider, iterations=50, warmup=10)

    assert result is not None
    assert result.provider == provider
    assert result.iterations == 50
    assert result.mean_ms > 0
    assert result.min_ms <= result.mean_ms <= result.max_ms
    assert result.min_ms <= result.median_ms <= result.max_ms
    print(f"[OK] {provider}: mean={result.mean_ms:.4f}ms over {result.iterations} iterations")


def test_bench_returns_none_for_missing_provider():
    from trt_ladder import ortbench

    if not ortbench.HAVE_ORT:
        print("[SKIP] onnxruntime not installed")
        return

    _ensure_model()
    result = ortbench.bench(MODEL_PATH, provider="ThisProviderDoesNotExist", iterations=5)
    assert result is None


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"[PASS] {name}")
        except AssertionError as exc:
            failures += 1
            print(f"[FAIL] {name}: {exc}")
        except Exception as exc:
            failures += 1
            print(f"[ERROR] {name}: {type(exc).__name__}: {exc}")
    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
