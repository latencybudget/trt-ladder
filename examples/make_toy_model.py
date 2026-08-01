"""Build a tiny, synthetic ONNX model for smoke-testing the ORT comparison path.

No trained weights, no proprietary architecture -- just enough graph (two
Gemm layers with a ReLU) to be a legal ONNX model that `ortbench.bench()` can
load and time. This lets the ONNX Runtime side of trt-ladder be verified on
any machine, without a Jetson or TensorRT.

Run: python examples/make_toy_model.py
"""

from __future__ import annotations

import os

import numpy as np
import onnx
from onnx import TensorProto, helper


def build(path: str, in_dim: int = 64, hidden: int = 128, out_dim: int = 10) -> None:
    rng = np.random.default_rng(0)
    w1 = rng.standard_normal((in_dim, hidden)).astype(np.float32)
    b1 = rng.standard_normal((hidden,)).astype(np.float32)
    w2 = rng.standard_normal((hidden, out_dim)).astype(np.float32)
    b2 = rng.standard_normal((out_dim,)).astype(np.float32)

    def initializer(name: str, arr: np.ndarray) -> onnx.TensorProto:
        return helper.make_tensor(name, TensorProto.FLOAT, arr.shape, arr.flatten().tolist())

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", in_dim])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", out_dim])

    gemm1 = helper.make_node("Gemm", ["input", "w1", "b1"], ["h1"], name="gemm1")
    relu = helper.make_node("Relu", ["h1"], ["h1_act"], name="relu")
    gemm2 = helper.make_node("Gemm", ["h1_act", "w2", "b2"], ["output"], name="gemm2")

    graph = helper.make_graph(
        [gemm1, relu, gemm2],
        "toy_mlp",
        [x],
        [y],
        initializer=[
            initializer("w1", w1),
            initializer("b1", b1),
            initializer("w2", w2),
            initializer("b2", b2),
        ],
    )
    model = helper.make_model(graph, producer_name="trt-ladder-example")
    model.opset_import[0].version = 13
    # Pin IR version explicitly: newer `onnx` packages default to a higher IR
    # version than older onnxruntime builds accept ("Unsupported model IR
    # version"). IR 8 pairs with opset 13 and is broadly compatible.
    model.ir_version = 8
    onnx.checker.check_model(model)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    onnx.save(model, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    build(os.path.join(os.path.dirname(__file__), "toy_mlp.onnx"))
