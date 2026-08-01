"""trt-ladder: TensorRT precision ladder profiler for NVIDIA Jetson.

Answers two questions that standard benchmarking misses:

  1. Which precision is actually fastest for *this* model? (Not always INT8.)
  2. How much of the latency is the model, and how much is the framework
     wrapped around it? (Often most of it, on small models.)
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
