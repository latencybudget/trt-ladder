"""Build and benchmark TensorRT engines through trtexec.

trtexec is used rather than a Python TensorRT binding on purpose: it reports
*GPU Compute Time* separately from end-to-end latency, which is the only way to
tell a slow model apart from a slow wrapper around a fast model. That
distinction is the point of this tool.

Precision modes map to trtexec flags as follows:

    fp32        (no flag)      baseline
    fp16        --fp16
    int8        --int8         pure INT8; frequently SLOWER than fp32, see README
    int8_fp16   --int8 --fp16  mixed; TRT picks per-layer, usually the winner

`int8` without a calibration cache makes trtexec calibrate on random data. That
produces meaningless quantisation ranges, so it is a latency probe only -- never
read accuracy off an engine built that way.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import Iterable, Optional

PRECISIONS = ("fp32", "fp16", "int8", "int8_fp16")

_PRECISION_FLAGS = {
    "fp32": [],
    "fp16": ["--fp16"],
    "int8": ["--int8"],
    "int8_fp16": ["--int8", "--fp16"],
}

PRECISION_LABELS = {
    "fp32": "FP32",
    "fp16": "FP16",
    "int8": "INT8 (pure)",
    "int8_fp16": "INT8+FP16 (mixed)",
}

# "min = 0.857 ms, max = 0.979 ms, mean = 0.912 ms, median = 0.916 ms, percentile(99%) = 0.970 ms"
_STAT_RE = re.compile(
    r"min\s*=\s*(?P<min>[\d.]+)\s*ms.*?"
    r"max\s*=\s*(?P<max>[\d.]+)\s*ms.*?"
    r"mean\s*=\s*(?P<mean>[\d.]+)\s*ms.*?"
    r"median\s*=\s*(?P<median>[\d.]+)\s*ms"
    r"(?:.*?percentile\((?P<pct>[\d.]+)%\)\s*=\s*(?P<p>[\d.]+)\s*ms)?",
    re.IGNORECASE,
)

# Strip the "[03/14/2026-11:02:31] [I] " prefix trtexec puts on every line.
_LINE_RE = re.compile(r"^(?:\[[^\]]*\]\s*)*\[I\]\s*(?P<key>[A-Za-z0-9 /()%_-]+?):\s*(?P<rest>.*)$")

_THROUGHPUT_RE = re.compile(r"([\d.]+)\s*qps", re.IGNORECASE)


class TrtexecError(RuntimeError):
    """trtexec exited non-zero, or produced no parseable performance summary."""


@dataclass
class Stat:
    min: float
    max: float
    mean: float
    median: float
    p99: Optional[float] = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchResult:
    precision: str
    gpu_compute: Optional[Stat] = None
    latency: Optional[Stat] = None
    h2d: Optional[Stat] = None
    d2h: Optional[Stat] = None
    throughput_qps: Optional[float] = None
    engine_bytes: Optional[int] = None
    build_seconds: Optional[float] = None
    log_path: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "precision": self.precision,
            "label": PRECISION_LABELS.get(self.precision, self.precision),
            "gpu_compute": self.gpu_compute.as_dict() if self.gpu_compute else None,
            "latency": self.latency.as_dict() if self.latency else None,
            "h2d": self.h2d.as_dict() if self.h2d else None,
            "d2h": self.d2h.as_dict() if self.d2h else None,
            "throughput_qps": self.throughput_qps,
            "engine_bytes": self.engine_bytes,
            "build_seconds": self.build_seconds,
            "log_path": self.log_path,
        }


def _parse_stat(rest: str) -> Optional[Stat]:
    m = _STAT_RE.search(rest)
    if not m:
        return None
    p = m.group("p")
    return Stat(
        min=float(m.group("min")),
        max=float(m.group("max")),
        mean=float(m.group("mean")),
        median=float(m.group("median")),
        p99=float(p) if p else None,
    )


def parse_trtexec_output(text: str) -> dict:
    """Pull the performance summary out of a trtexec log.

    Only the last occurrence of each key is kept: trtexec prints an inference
    summary and then a performance summary, and the latter is the one with the
    percentile fields populated.
    """
    out: dict = {}
    for line in text.splitlines():
        m = _LINE_RE.match(line.strip())
        if not m:
            continue
        key = m.group("key").strip().lower()
        rest = m.group("rest").strip()

        if key == "throughput":
            t = _THROUGHPUT_RE.search(rest)
            if t:
                out["throughput_qps"] = float(t.group(1))
            continue

        target = {
            "gpu compute time": "gpu_compute",
            "latency": "latency",
            "h2d latency": "h2d",
            "d2h latency": "d2h",
        }.get(key)
        if not target:
            continue
        stat = _parse_stat(rest)
        if stat is not None:
            out[target] = stat
    return out


def _shape_args(shapes: Optional[str], dynamic: bool) -> list[str]:
    """Translate a shape spec into trtexec flags.

    A single --shapes is enough for a static-profile build; dynamic models need
    the min/opt/max triple at build time and --shapes at inference time.
    """
    if not shapes:
        return []
    if not dynamic:
        return [f"--shapes={shapes}"]
    return [
        f"--minShapes={shapes}",
        f"--optShapes={shapes}",
        f"--maxShapes={shapes}",
        f"--shapes={shapes}",
    ]


def run_one(
    onnx_path: str,
    precision: str,
    workdir: str,
    trtexec: str,
    iterations: int = 1000,
    warmup_ms: int = 500,
    shapes: Optional[str] = None,
    dynamic: bool = False,
    extra_args: Optional[Iterable[str]] = None,
    timeout: float = 1800.0,
) -> BenchResult:
    """Build an engine at `precision` and benchmark it in a single trtexec call.

    Raises TrtexecError on a non-zero exit or an unparseable log; the caller
    decides whether one failed rung aborts the ladder.
    """
    if precision not in _PRECISION_FLAGS:
        raise ValueError(f"unknown precision {precision!r}; expected one of {PRECISIONS}")
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(onnx_path)

    os.makedirs(workdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    engine_path = os.path.join(workdir, f"{stem}.{precision}.engine")
    log_path = os.path.join(workdir, f"{stem}.{precision}.log")

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--iterations={iterations}",
        f"--warmUp={warmup_ms}",
        # Spin-wait removes the host-side sleep/wake jitter that otherwise
        # shows up as a fat tail on sub-millisecond models.
        "--useSpinWait",
        "--percentile=99",
        "--noDataTransfers=false",
    ]
    cmd += _PRECISION_FLAGS[precision]
    cmd += _shape_args(shapes, dynamic)
    if extra_args:
        cmd += list(extra_args)

    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise TrtexecError(
            f"{precision}: trtexec exceeded {timeout:.0f}s. INT8 builds on Orin "
            "can legitimately take several minutes; raise --timeout if needed."
        ) from exc
    elapsed = time.monotonic() - started

    combined = (proc.stdout or "") + (proc.stderr or "")
    try:
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(" ".join(cmd) + "\n\n" + combined)
    except OSError:
        log_path = None

    if proc.returncode != 0:
        tail = "\n".join(combined.strip().splitlines()[-15:])
        raise TrtexecError(f"{precision}: trtexec exited {proc.returncode}\n{tail}")

    parsed = parse_trtexec_output(combined)
    if "gpu_compute" not in parsed:
        tail = "\n".join(combined.strip().splitlines()[-15:])
        raise TrtexecError(f"{precision}: no performance summary in trtexec output\n{tail}")

    engine_bytes = None
    try:
        engine_bytes = os.path.getsize(engine_path)
    except OSError:
        pass

    return BenchResult(
        precision=precision,
        gpu_compute=parsed.get("gpu_compute"),
        latency=parsed.get("latency"),
        h2d=parsed.get("h2d"),
        d2h=parsed.get("d2h"),
        throughput_qps=parsed.get("throughput_qps"),
        engine_bytes=engine_bytes,
        build_seconds=elapsed,
        log_path=log_path,
    )
