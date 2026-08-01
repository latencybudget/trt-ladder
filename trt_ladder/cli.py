"""Command-line entry point: `trt-ladder model.onnx`.

Runs the full precision ladder, optionally compares against ONNX Runtime, and
writes a Markdown + JSON report. A rung that fails to build does not abort the
run -- a partial ladder with the failure recorded is more useful than nothing,
and some models legitimately cannot build at every precision.
"""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from . import env, ortbench, report
from .trtexec import PRECISIONS, TrtexecError, run_one


def _parse_dim_overrides(spec: str | None) -> dict:
    """`batch=1,seq=128` -> {"batch": 1, "seq": 128}"""
    if not spec:
        return {}
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(f"expected name=value, got {part!r}")
        name, _, value = part.partition("=")
        try:
            out[name.strip()] = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{part!r}: value must be an integer") from exc
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trt-ladder",
        description=(
            "Benchmark an ONNX model across TensorRT precisions and separate "
            "GPU compute from framework overhead."
        ),
    )
    p.add_argument("model", help="path to the .onnx model")
    p.add_argument(
        "-o", "--outdir", default="trt_ladder_out",
        help="directory for engines, logs and the report (default: %(default)s)",
    )
    p.add_argument(
        "-p", "--precisions", default="fp32,fp16,int8,int8_fp16",
        help="comma-separated subset of: " + ",".join(PRECISIONS),
    )
    p.add_argument("-i", "--iterations", type=int, default=1000,
                   help="timed trtexec iterations per precision (default: %(default)s)")
    p.add_argument("--warmup-ms", type=int, default=500,
                   help="trtexec warm-up in ms (default: %(default)s)")
    p.add_argument("--shapes", default=None,
                   help="trtexec shape spec, e.g. 'input:1x3x224x224'")
    p.add_argument("--dynamic", action="store_true",
                   help="model has dynamic axes; emit min/opt/max shape flags")
    p.add_argument("--trtexec", default=None, help="explicit path to trtexec")
    p.add_argument("--timeout", type=float, default=1800.0,
                   help="per-precision trtexec timeout in seconds (default: %(default)s)")
    p.add_argument("--no-ort", action="store_true",
                   help="skip the ONNX Runtime comparison")
    p.add_argument("--ort-provider", default="CUDAExecutionProvider",
                   help="ORT execution provider (default: %(default)s)")
    p.add_argument("--ort-iterations", type=int, default=300,
                   help="ORT timed iterations (default: %(default)s)")
    p.add_argument("--dim-overrides", default=None,
                   help="concrete sizes for symbolic ORT input dims, e.g. 'batch=1,seq=128'")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.isfile(args.model):
        print(f"[ERROR] model not found: {args.model}", file=sys.stderr)
        return 2

    requested = [x.strip() for x in args.precisions.split(",") if x.strip()]
    unknown = [x for x in requested if x not in PRECISIONS]
    if unknown:
        print(f"[ERROR] unknown precision(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"        expected a subset of: {', '.join(PRECISIONS)}", file=sys.stderr)
        return 2
    if not requested:
        print("[ERROR] no precisions selected", file=sys.stderr)
        return 2

    try:
        dim_overrides = _parse_dim_overrides(args.dim_overrides)
    except argparse.ArgumentTypeError as exc:
        print(f"[ERROR] --dim-overrides: {exc}", file=sys.stderr)
        return 2

    platform = env.probe(args.trtexec)
    if not platform.trtexec:
        print("[ERROR] trtexec not found. Pass --trtexec /path/to/trtexec.", file=sys.stderr)
        print("        On JetPack it is usually /usr/src/tensorrt/bin/trtexec", file=sys.stderr)
        return 3

    for w in platform.warnings():
        print(f"[WARN] {w}")

    os.makedirs(args.outdir, exist_ok=True)
    engines_dir = os.path.join(args.outdir, "engines")

    results = []
    failures: list[tuple[str, str]] = []
    for precision in requested:
        print(f"[INFO] building + benchmarking {precision} ...", flush=True)
        try:
            res = run_one(
                onnx_path=args.model,
                precision=precision,
                workdir=engines_dir,
                trtexec=platform.trtexec,
                iterations=args.iterations,
                warmup_ms=args.warmup_ms,
                shapes=args.shapes,
                dynamic=args.dynamic,
                timeout=args.timeout,
            )
        except (TrtexecError, FileNotFoundError, ValueError) as exc:
            print(f"[ERROR] {precision} failed: {exc}", file=sys.stderr)
            failures.append((precision, str(exc)))
            continue
        gc = res.gpu_compute
        print(f"[OK]   {precision}: GPU compute mean {gc.mean:.3f} ms" if gc else f"[OK]   {precision}")
        results.append(res)

    if not results:
        print("[ERROR] every precision failed; nothing to report.", file=sys.stderr)
        return 1

    ort_result = None
    if not args.no_ort:
        if not ortbench.HAVE_ORT:
            print("[WARN] onnxruntime not installed; skipping framework-overhead comparison.")
        elif args.ort_provider not in ortbench.available_providers():
            have = ", ".join(ortbench.available_providers()) or "none"
            print(f"[WARN] provider {args.ort_provider} unavailable (have: {have}); skipping.")
        else:
            print(f"[INFO] benchmarking ONNX Runtime ({args.ort_provider}) ...", flush=True)
            try:
                ort_result = ortbench.bench(
                    args.model,
                    provider=args.ort_provider,
                    iterations=args.ort_iterations,
                    dim_overrides=dim_overrides,
                )
            except Exception as exc:  # ORT raises a wide range of backend errors
                print(f"[WARN] ORT benchmark failed: {exc}")

    md = report.render_markdown(
        args.model, platform, results, ort_result, iterations=args.iterations
    )
    if failures:
        md += "\n## Failed builds\n\n"
        for precision, msg in failures:
            first = msg.strip().splitlines()[0] if msg.strip() else "unknown error"
            md += f"- `{precision}`: {first}\n"
        md += (
            "\nA failed rung is a result too: it tells the next person not to spend "
            "a day on that path.\n"
        )

    md_path = os.path.join(args.outdir, "report.md")
    json_path = os.path.join(args.outdir, "report.json")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md)
    with open(json_path, "w", encoding="utf-8") as fh:
        fh.write(report.render_json(args.model, platform, results, ort_result))

    print(f"\n[OK] report written to {md_path}")
    print(f"[OK] raw data written to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
