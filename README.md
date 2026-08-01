# trt-ladder

**Profile an ONNX model across every TensorRT precision on Jetson, and find out how much of your latency isn't the model at all.**

Two things bite people deploying to Jetson, and both look like bugs when you hit them:

1. **INT8 can be slower than FP32.** Not sometimes-in-theory — routinely, on small or attention-heavy models.
2. **Most of your measured latency may not be GPU work.** On a sub-millisecond model, the Python calling path can cost more than the network.

Neither shows up if you benchmark the way most people do (wrap `session.run()` in a timer and read the mean). `trt-ladder` measures both, and writes a report you can hand to someone who wasn't in the room.

---

## What it does

```bash
trt-ladder model.onnx --shapes 'input:1x3x224x224'
```

Builds and benchmarks the full precision ladder — FP32, FP16, pure INT8, mixed INT8+FP16 — then compares the best of them against what ONNX Runtime actually delivers to a Python caller.

Output is `report.md` + `report.json`, with the raw `trtexec` logs kept alongside the engines for audit.

### Example: the precision ladder

| Precision | Mean (ms) | Median (ms) | P99 (ms) | vs FP32 | Throughput (qps) | Engine |
|---|---|---|---|---|---|---|
| FP32 | 0.912 | 0.916 | 0.970 | 1.00x | 1049 | 1.70 MB |
| FP16 | 0.811 | 0.810 | 0.841 | 1.12x (-11.1%) | 1175 | 983 KB |
| INT8 (pure) | 1.082 | 1.079 | 1.140 | 0.84x (+18.6%) | 924 | 3.50 MB |
| INT8+FP16 (mixed) | 0.801 | 0.800 | 0.828 | 1.14x (-12.2%) | 1190 | 992 KB |

> **Pure INT8 is 18.6% SLOWER than FP32 here.** This is a normal result on small or attention-heavy models, not a misconfiguration: per-layer quantise/dequantise nodes cost more than the narrower arithmetic saves, and precision-sensitive ops (Softmax, LayerNorm) fall back to a wider type while keeping their Q/DQ pairs.

Note the engine size too — pure INT8 came out **twice as large as FP32**, because it's carrying calibration tables and fallback layers. That's the tell.

### Example: where the time actually goes

| Path | Mean (ms) | What it measures |
|---|---|---|
| ONNX Runtime (CUDAExecutionProvider) | 2.468 | end-to-end `session.run()` from Python |
| TensorRT native (trtexec) | 0.801 | GPU compute only, best precision |
| **Framework overhead** | **1.667** | **68% of the Python-visible latency** |

**This is usually the finding that matters.** If two thirds of your latency is session dispatch and allocator traffic, another week of model optimisation buys you nothing. The win is in the calling path.

---

## Install

On the Jetson:

```bash
pip install -e .
```

`trtexec` ships with JetPack at `/usr/src/tensorrt/bin/trtexec` and is found automatically. For the framework-overhead comparison you also need `onnxruntime-gpu` built for your JetPack — if it's missing, the ladder still runs and the report just omits that section.

---

## Usage

```bash
# Full ladder
trt-ladder model.onnx

# Static shapes
trt-ladder model.onnx --shapes 'input:1x3x224x224'

# Dynamic axes: emits min/opt/max at build time
trt-ladder model.onnx --shapes 'input:1x3x224x224' --dynamic

# Skip INT8 if you only care about the FP16 decision
trt-ladder model.onnx -p fp32,fp16

# Symbolic dims for the ORT side
trt-ladder model.onnx --dim-overrides 'batch=1,seq=128'
```

Everything lands in `--outdir` (default `trt_ladder_out/`): `report.md`, `report.json`, and `engines/` with one `.engine` and one `.log` per precision.

---

## Why the numbers are trustworthy

Benchmark hygiene is most of the value here. `trt-ladder` enforces it and, more importantly, **tells you when it couldn't**:

- **Clock state is checked and reported.** If `jetson_clocks` hasn't been run, the report says so at the top. Unlocked clocks produce run-to-run variance that swamps a 10% effect.
- **GPU compute is read from `trtexec`'s performance summary**, not from a host-side wall clock — that's the only way to separate compute from dispatch.
- **`--useSpinWait`** removes host sleep/wake jitter, which otherwise shows up as a fat tail on sub-millisecond models.
- **Median and P99 are reported alongside the mean.** A mean alone hides the tail, and the tail is what makes a control loop miss its deadline.
- **Power mode and L4T version are recorded**, so the run can be reproduced.
- **Failed builds are reported, not silently dropped.** A rung that won't build is a result: it saves the next person a day.

---

## Limitations — read these

- **Pure-INT8 engines are built with random calibration.** trtexec's default calibrator sees noise, so the quantisation ranges are meaningless. These engines are a **latency probe only** — never read accuracy off one. Proper INT8 accuracy needs a calibration cache built from real data.
- **`trtexec` measures the engine, not your pipeline.** Pre-processing, buffer management and I/O are yours to measure.
- **Random input data.** Fine for latency (TensorRT's timing doesn't depend on values), useless for accuracy.
- **The ORT comparison is one configuration** — single-threaded intra-op, one provider. It's a demonstration that the gap exists, not a full study of your serving stack.
- Tested against TensorRT 8.5.x on JetPack 5.1.x. Newer TRT changes `trtexec` output formatting; the parser is defensive but not clairvoyant.

---

## What to do with the results

| Finding | What it means |
|---|---|
| FP16 ≈ FP32 | You're not compute-bound. Look at the calling path or the data pipeline. |
| Pure INT8 slower than FP16 | Expected on small/attention models. Use mixed `--int8 --fp16` and move on. |
| Framework overhead > 50% | Stop optimising the model. Batch, or move to a native runtime. |
| H2D/D2H flat across precisions | Compute-bound. Zero-copy won't help you. |
| H2D/D2H dominant | Transfer-bound. Zero-copy / mapped host memory will. |
| Huge P99 vs median | Something is preempting you. Check clocks, thermals, and other processes. |

---

## License

MIT
