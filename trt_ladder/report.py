"""Render a run into Markdown and JSON.

The report is written to be handed to someone who was not in the room: it
states the clock and power state the numbers were taken under, flags the two
results people most often misread (INT8 regression, framework overhead), and
keeps the raw trtexec logs referenced by path.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from .trtexec import BenchResult, PRECISION_LABELS

_NA = "--"


def _fmt(value: Optional[float], digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return _NA
    return f"{value:.{digits}f}{suffix}"


def _fmt_bytes(n: Optional[int]) -> str:
    if n is None:
        return _NA
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MB"
    return f"{n / 1024:.0f} KB"


def _speedup(baseline: Optional[float], value: Optional[float]) -> str:
    if not baseline or not value or value <= 0:
        return _NA
    if value == baseline:
        return "1.00x (baseline)"
    ratio = baseline / value
    delta = (value - baseline) / baseline * 100.0
    sign = "+" if delta >= 0 else ""
    return f"{ratio:.2f}x ({sign}{delta:.1f}%)"


def ladder_rows(results: list[BenchResult]) -> list[list[str]]:
    baseline = None
    for r in results:
        if r.precision == "fp32" and r.gpu_compute:
            baseline = r.gpu_compute.mean
            break

    rows = []
    for r in results:
        gc = r.gpu_compute
        rows.append(
            [
                PRECISION_LABELS.get(r.precision, r.precision),
                _fmt(gc.mean if gc else None),
                _fmt(gc.median if gc else None),
                _fmt(gc.p99 if gc else None),
                _speedup(baseline, gc.mean if gc else None),
                _fmt(r.throughput_qps, 0),
                _fmt_bytes(r.engine_bytes),
            ]
        )
    return rows


def transfer_rows(results: list[BenchResult]) -> list[list[str]]:
    rows = []
    for r in results:
        h2d = r.h2d.mean if r.h2d else None
        d2h = r.d2h.mean if r.d2h else None
        gc = r.gpu_compute.mean if r.gpu_compute else None
        total = None
        if gc is not None:
            total = gc + (h2d or 0.0) + (d2h or 0.0)
        rows.append(
            [
                PRECISION_LABELS.get(r.precision, r.precision),
                _fmt(h2d, 4),
                _fmt(d2h, 4),
                _fmt(gc),
                _fmt(total),
            ]
        )
    return rows


def overhead_rows(results: list[BenchResult], ort_result) -> tuple[list[list[str]], Optional[float]]:
    """Compare the Python-visible latency against native GPU compute.

    Returns (rows, overhead_fraction). The fraction is the share of the
    end-to-end Python call that is not GPU work -- the number worth putting in
    front of whoever asked why the model is slow.
    """
    if ort_result is None:
        return [], None

    fastest = None
    for r in results:
        if r.gpu_compute and (fastest is None or r.gpu_compute.mean < fastest):
            fastest = r.gpu_compute.mean
    if not fastest or ort_result.mean_ms <= 0:
        return [], None

    overhead = ort_result.mean_ms - fastest
    fraction = overhead / ort_result.mean_ms if ort_result.mean_ms > 0 else None

    rows = [
        [f"ONNX Runtime ({ort_result.provider})", _fmt(ort_result.mean_ms), "end-to-end `session.run()` from Python"],
        ["TensorRT native (trtexec)", _fmt(fastest), "GPU compute only, best precision"],
        [
            "**Framework overhead**",
            f"**{_fmt(overhead)}**",
            f"**{fraction * 100:.0f}% of the Python-visible latency**" if fraction is not None else _NA,
        ],
    ]
    return rows, fraction


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No data._\n"
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


def _int8_note(results: list[BenchResult]) -> Optional[str]:
    """Call out the INT8-is-slower case explicitly; it reads as a bug otherwise."""
    by = {r.precision: r for r in results}
    fp32 = by.get("fp32")
    pure = by.get("int8")
    if not (fp32 and pure and fp32.gpu_compute and pure.gpu_compute):
        return None
    if pure.gpu_compute.mean <= fp32.gpu_compute.mean:
        return None
    delta = (pure.gpu_compute.mean - fp32.gpu_compute.mean) / fp32.gpu_compute.mean * 100.0
    mixed = by.get("int8_fp16")
    tail = ""
    if mixed and mixed.gpu_compute and fp32.gpu_compute:
        tail = (
            f" Mixed INT8+FP16 came in at {mixed.gpu_compute.mean:.3f} ms, so the "
            "regression is specific to forcing INT8 everywhere, not to "
            "quantisation as such."
        )
    return (
        f"**Pure INT8 is {delta:.1f}% SLOWER than FP32 here.** This is a normal "
        "result on small or attention-heavy models, not a misconfiguration: "
        "per-layer quantise/dequantise nodes cost more than the narrower "
        "arithmetic saves, and precision-sensitive ops (Softmax, LayerNorm) "
        "fall back to a wider type while keeping their Q/DQ pairs." + tail
    )


def render_markdown(
    onnx_path: str,
    platform,
    results: list[BenchResult],
    ort_result=None,
    iterations: int = 1000,
) -> str:
    p = platform
    lines: list[str] = []
    a = lines.append

    a(f"# TRT precision ladder -- `{os.path.basename(onnx_path)}`\n")

    warns = p.warnings()
    if warns:
        a("> **Measurement conditions to check before quoting these numbers**\n>")
        for w in warns:
            a(f"> - {w}")
        a("")

    a("## Environment\n")
    clock = p.gpu_clock
    locked = {True: "locked at max", False: "NOT locked", None: "unknown"}[clock.locked]
    a(
        _table(
            ["Item", "Value"],
            [
                ["Device", p.model or _NA],
                ["L4T", p.l4t or _NA],
                ["TensorRT", p.trt_version or _NA],
                ["Power mode", p.power_mode or _NA],
                ["GPU clock", locked],
                ["trtexec iterations", str(iterations)],
            ],
        )
    )

    a("\n## Precision ladder (GPU compute time)\n")
    a(
        _table(
            ["Precision", "Mean (ms)", "Median (ms)", "P99 (ms)", "vs FP32", "Throughput (qps)", "Engine"],
            ladder_rows(results),
        )
    )

    note = _int8_note(results)
    if note:
        a("\n" + note + "\n")

    rows, fraction = overhead_rows(results, ort_result)
    if rows:
        a("\n## Where the time actually goes\n")
        a(_table(["Path", "Mean (ms)", "What it measures"], rows))
        if fraction is not None and fraction >= 0.3:
            a(
                f"\n**{fraction * 100:.0f}% of the latency a Python caller sees is not GPU work.** "
                "Optimising the model further will not move that part; it comes from "
                "session dispatch, allocation and provider plumbing on every call. "
                "On this model the win is in the calling path, not the network.\n"
            )

    a("\n## Transfer breakdown\n")
    a(
        _table(
            ["Precision", "H2D (ms)", "D2H (ms)", "GPU compute (ms)", "Total (ms)"],
            transfer_rows(results),
        )
    )
    a(
        "\nIf H2D/D2H are flat across precisions, the bottleneck is compute, not "
        "the bus -- and zero-copy will not help. If they dominate, it will.\n"
    )

    a("\n## Method\n")
    a(
        "- `trtexec` with `--useSpinWait` and `--percentile=99`; GPU compute time is\n"
        "  read from the performance summary, not wall clock.\n"
        f"- {iterations} timed iterations after a 500 ms warm-up per precision.\n"
        "- Pure-INT8 engines are built with random calibration and are a **latency\n"
        "  probe only** -- do not read accuracy off them.\n"
        "- Raw trtexec logs are kept next to the engines for audit.\n"
    )

    return "\n".join(lines)


def render_json(onnx_path: str, platform, results: list[BenchResult], ort_result=None) -> str:
    payload = {
        "model": os.path.abspath(onnx_path),
        "platform": platform.as_dict(),
        "warnings": platform.warnings(),
        "results": [r.as_dict() for r in results],
        "onnxruntime": ort_result.as_dict() if ort_result else None,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
