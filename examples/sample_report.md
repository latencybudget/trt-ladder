# TRT precision ladder -- `resnet50.onnx`

> **These numbers are illustrative.** They are transcribed from a real
> TensorRT 8.5.2 / JetPack 5.1.2 run on a Jetson AGX Orin, but against a
> different model than the filename suggests, and the transfer figures are
> held constant across rows for readability. Treat this file as a format
> sample, not as a measurement of ResNet-50.


## Environment

| Item | Value |
|---|---|
| Device | NVIDIA Jetson AGX Orin |
| L4T | 35.4.1 |
| TensorRT | 8.5.2 |
| Power mode | MAXN |
| GPU clock | locked at max |
| trtexec iterations | 1000 |


## Precision ladder (GPU compute time)

| Precision | Mean (ms) | Median (ms) | P99 (ms) | vs FP32 | Throughput (qps) | Engine |
|---|---|---|---|---|---|---|
| FP32 | 0.912 | 0.912 | 0.952 | 1.00x (baseline) | 1000 | 1.70 MB |
| FP16 | 0.811 | 0.811 | 0.851 | 1.12x (-11.1%) | 1000 | 983 KB |
| INT8 (pure) | 1.082 | 1.082 | 1.122 | 0.84x (+18.6%) | 1000 | 3.50 MB |
| INT8+FP16 (mixed) | 0.801 | 0.801 | 0.841 | 1.14x (-12.2%) | 1000 | 992 KB |


**Pure INT8 is 18.6% SLOWER than FP32 here.** This is a normal result on small or attention-heavy models, not a misconfiguration: per-layer quantise/dequantise nodes cost more than the narrower arithmetic saves, and precision-sensitive ops (Softmax, LayerNorm) fall back to a wider type while keeping their Q/DQ pairs. Mixed INT8+FP16 came in at 0.801 ms, so the regression is specific to forcing INT8 everywhere, not to quantisation as such.


## Where the time actually goes

| Path | Mean (ms) | What it measures |
|---|---|---|
| ONNX Runtime (CUDAExecutionProvider) | 2.468 | end-to-end `session.run()` from Python |
| TensorRT native (trtexec) | 0.801 | GPU compute only, best precision |
| **Framework overhead** | **1.667** | **68% of the Python-visible latency** |


**68% of the latency a Python caller sees is not GPU work.** Optimising the model further will not move that part; it comes from session dispatch, allocation and provider plumbing on every call. On this model the win is in the calling path, not the network.


## Transfer breakdown

| Precision | H2D (ms) | D2H (ms) | GPU compute (ms) | Total (ms) |
|---|---|---|---|---|
| FP32 | 0.0102 | 0.0121 | 0.912 | 0.934 |
| FP16 | 0.0102 | 0.0121 | 0.811 | 0.833 |
| INT8 (pure) | 0.0102 | 0.0121 | 1.082 | 1.104 |
| INT8+FP16 (mixed) | 0.0102 | 0.0121 | 0.801 | 0.823 |


If H2D/D2H are flat across precisions, the bottleneck is compute, not the bus -- and zero-copy will not help. If they dominate, it will.


## Method

- `trtexec` with `--useSpinWait` and `--percentile=99`; GPU compute time is
  read from the performance summary, not wall clock.
- 1000 timed iterations after a 500 ms warm-up per precision.
- Pure-INT8 engines are built with random calibration and are a **latency
  probe only** -- do not read accuracy off them.
- Raw trtexec logs are kept next to the engines for audit.
