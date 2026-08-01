"""Measure what a Python caller actually experiences, via ONNX Runtime.

trtexec reports what the GPU does. This module reports what your application
gets back, which on small models is frequently a very different number: session
dispatch, allocator traffic and provider plumbing are fixed per-call costs that
do not shrink as the model does.

The gap between the two is the headline finding of most profiling runs -- see
`report.overhead_rows`. Optional: if onnxruntime is missing, the ladder still
runs and the report simply omits the comparison.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

try:  # pragma: no cover - availability is environment-dependent
    import onnxruntime as ort

    HAVE_ORT = True
except ImportError:  # pragma: no cover
    ort = None
    HAVE_ORT = False


# ORT reports dynamic axes as either None or a string symbol; neither can be
# allocated, so substitute a concrete size.
_DEFAULT_DYNAMIC_DIM = 1

_ORT_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}


@dataclass
class OrtResult:
    provider: str
    mean_ms: float
    median_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    iterations: int

    def as_dict(self) -> dict:
        return asdict(self)


def _resolve_shape(shape, overrides: Optional[dict]) -> list[int]:
    out: list[int] = []
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            out.append(dim)
        elif isinstance(dim, str) and overrides and dim in overrides:
            out.append(int(overrides[dim]))
        else:
            out.append(_DEFAULT_DYNAMIC_DIM)
    return out


def _make_inputs(session, overrides: Optional[dict]) -> dict:
    feeds = {}
    for inp in session.get_inputs():
        dtype = _ORT_DTYPES.get(inp.type, np.float32)
        shape = _resolve_shape(inp.shape, overrides)
        if np.issubdtype(dtype, np.floating):
            arr = np.random.randn(*shape).astype(dtype)
        elif dtype is np.bool_:
            arr = np.zeros(shape, dtype=dtype)
        else:
            arr = np.zeros(shape, dtype=dtype)
        feeds[inp.name] = arr
    return feeds


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. Avoids a scipy/numpy-version dependency."""
    if not values:
        raise ValueError("empty sample")
    ordered = sorted(values)
    # nearest-rank: ceil(pct/100 * N), clamped into range
    idx = int(-(-len(ordered) * pct // 100)) - 1
    return ordered[max(0, min(idx, len(ordered) - 1))]


def available_providers() -> list[str]:
    if not HAVE_ORT:
        return []
    return list(ort.get_available_providers())


def bench(
    onnx_path: str,
    provider: str = "CUDAExecutionProvider",
    iterations: int = 300,
    warmup: int = 50,
    dim_overrides: Optional[dict] = None,
) -> Optional[OrtResult]:
    """Time `session.run()` end-to-end from Python.

    Returns None when onnxruntime is absent or the requested provider is not
    registered in this build -- a missing comparison is better than a
    misleading one taken on a different backend than the caller asked for.
    """
    if not HAVE_ORT:
        return None
    if provider not in available_providers():
        return None

    opts = ort.SessionOptions()
    # Keep ORT from silently multi-threading the CPU fallback, which would make
    # the wrapper overhead look smaller than it is on a loaded system.
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(onnx_path, sess_options=opts, providers=[provider])

    feeds = _make_inputs(session, dim_overrides)
    output_names = [o.name for o in session.get_outputs()]

    for _ in range(max(0, warmup)):
        session.run(output_names, feeds)

    samples: list[float] = []
    for _ in range(max(1, iterations)):
        t0 = time.perf_counter()
        session.run(output_names, feeds)
        samples.append((time.perf_counter() - t0) * 1000.0)

    return OrtResult(
        provider=provider,
        mean_ms=statistics.fmean(samples),
        median_ms=statistics.median(samples),
        p99_ms=_percentile(samples, 99),
        min_ms=min(samples),
        max_ms=max(samples),
        iterations=len(samples),
    )
