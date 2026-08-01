"""Parser and report tests using captured trtexec output.

The trtexec log format is the one external contract this tool depends on, so it
is pinned here with a real sample rather than trusted. The report assertions
cover the two findings the tool exists to surface: the INT8 regression callout
and the framework-overhead share.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trt_ladder.env import GpuClockState, Platform
from trt_ladder.report import ladder_rows, overhead_rows, render_markdown, _int8_note
from trt_ladder.trtexec import BenchResult, Stat, parse_trtexec_output

SAMPLE = """
[03/14/2026-11:02:29] [I] === Performance summary ===
[03/14/2026-11:02:29] [I] Throughput: 1049.23 qps
[03/14/2026-11:02:29] [I] Latency: min = 0.921 ms, max = 1.021 ms, mean = 0.934 ms, median = 0.933 ms, percentile(99%) = 1.002 ms
[03/14/2026-11:02:29] [I] Enqueue Time: min = 0.011 ms, max = 0.084 ms, mean = 0.019 ms, median = 0.017 ms, percentile(99%) = 0.062 ms
[03/14/2026-11:02:29] [I] H2D Latency: min = 0.0091 ms, max = 0.0223 ms, mean = 0.0102 ms, median = 0.0100 ms, percentile(99%) = 0.0180 ms
[03/14/2026-11:02:29] [I] GPU Compute Time: min = 0.857 ms, max = 0.979 ms, mean = 0.912 ms, median = 0.916 ms, percentile(99%) = 0.970 ms
[03/14/2026-11:02:29] [I] D2H Latency: min = 0.0105 ms, max = 0.0288 ms, mean = 0.0121 ms, median = 0.0118 ms, percentile(99%) = 0.0201 ms
[03/14/2026-11:02:29] [I] Total Host Walltime: 0.953 s
"""


def test_parse_gpu_compute():
    out = parse_trtexec_output(SAMPLE)
    gc = out["gpu_compute"]
    assert abs(gc.mean - 0.912) < 1e-9
    assert abs(gc.median - 0.916) < 1e-9
    assert abs(gc.min - 0.857) < 1e-9
    assert abs(gc.max - 0.979) < 1e-9
    assert abs(gc.p99 - 0.970) < 1e-9


def test_parse_transfers_and_throughput():
    out = parse_trtexec_output(SAMPLE)
    assert abs(out["h2d"].mean - 0.0102) < 1e-9
    assert abs(out["d2h"].mean - 0.0121) < 1e-9
    assert abs(out["throughput_qps"] - 1049.23) < 1e-9
    # "Latency" must not be confused with "GPU Compute Time"
    assert abs(out["latency"].mean - 0.934) < 1e-9


def test_parse_empty_is_not_a_crash():
    assert parse_trtexec_output("") == {}
    assert parse_trtexec_output("garbage\nmore garbage") == {}


def _result(precision, mean, engine_bytes=None):
    return BenchResult(
        precision=precision,
        gpu_compute=Stat(min=mean - 0.05, max=mean + 0.05, mean=mean, median=mean, p99=mean + 0.04),
        h2d=Stat(min=0.009, max=0.02, mean=0.0102, median=0.010),
        d2h=Stat(min=0.010, max=0.028, mean=0.0121, median=0.0118),
        throughput_qps=1000.0,
        engine_bytes=engine_bytes,
    )


LADDER = [
    _result("fp32", 0.912, 1_782_579),
    _result("fp16", 0.811, 1_006_632),
    _result("int8", 1.082, 3_670_016),
    _result("int8_fp16", 0.801, 1_015_808),
]


def test_ladder_speedup_direction():
    rows = ladder_rows(LADDER)
    by_label = {r[0]: r for r in rows}
    assert by_label["FP32"][4].startswith("1.00x")
    # FP16 is faster -> ratio > 1, percentage delta negative
    assert by_label["FP16"][4].startswith("1.12x")
    assert "-11.1%" in by_label["FP16"][4]
    # Pure INT8 is slower -> ratio < 1, percentage delta positive
    assert by_label["INT8 (pure)"][4].startswith("0.84x")
    assert "+18.6%" in by_label["INT8 (pure)"][4]


def test_int8_regression_is_called_out():
    note = _int8_note(LADDER)
    assert note is not None
    assert "SLOWER" in note
    assert "18.6%" in note


def test_no_int8_note_when_int8_is_faster():
    fine = [_result("fp32", 1.000), _result("int8", 0.700)]
    assert _int8_note(fine) is None


def test_missing_fp32_baseline_degrades_gracefully():
    rows = ladder_rows([_result("fp16", 0.811)])
    assert rows[0][4] == "--"


class _Ort:
    provider = "CUDAExecutionProvider"
    mean_ms = 2.468


def test_overhead_fraction():
    rows, fraction = overhead_rows(LADDER, _Ort())
    assert fraction is not None
    # (2.468 - 0.801) / 2.468
    assert abs(fraction - 0.6755) < 1e-3
    assert "68%" in rows[2][2]


def test_overhead_without_ort_is_empty():
    rows, fraction = overhead_rows(LADDER, None)
    assert rows == []
    assert fraction is None


def _platform(locked=True):
    return Platform(
        model="NVIDIA Jetson AGX Orin",
        is_jetson=True,
        l4t="35.4.1",
        trt_version="8.5.2",
        trtexec="/usr/src/tensorrt/bin/trtexec",
        power_mode="MAXN",
        gpu_clock=GpuClockState(cur_hz=930_750_000, max_hz=930_750_000)
        if locked
        else GpuClockState(cur_hz=420_750_000, max_hz=930_750_000),
    )


def test_render_markdown_contains_key_sections():
    md = render_markdown("model.onnx", _platform(), LADDER, _Ort())
    assert "# TRT precision ladder" in md
    assert "Precision ladder (GPU compute time)" in md
    assert "Where the time actually goes" in md
    assert "Transfer breakdown" in md
    assert "SLOWER" in md
    assert "68% of the latency a Python caller sees is not GPU work" in md


def test_unlocked_clocks_produce_a_warning():
    p = _platform(locked=False)
    assert p.gpu_clock.locked is False
    warns = p.warnings()
    assert any("not locked" in w for w in warns)
    assert any("45%" in w for w in warns)  # 420750000 / 930750000
    md = render_markdown("model.onnx", p, LADDER, None)
    assert "jetson_clocks" in md


def test_unknown_clock_state_is_distinct_from_unlocked():
    p = _platform()
    p.gpu_clock = GpuClockState(cur_hz=None, max_hz=None)
    assert p.gpu_clock.locked is None
    assert any("unknown" in w for w in p.warnings())


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
