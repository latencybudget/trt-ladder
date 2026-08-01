---
name: A precision is slower than expected
about: Report a case where FP16 or INT8 came out slower than a lower-effort precision
title: "[slow] "
labels: precision-regression
---

**Model type** (e.g. CNN, ViT/transformer, small vs large):

**Precisions and GPU compute times from the ladder report:**

```
paste the "Precision ladder" table from report.md here
```

**Device / JetPack / TensorRT version:**

**Was `jetson_clocks` run before benchmarking?**

**Anything else you changed from the default (`--shapes`, `--dynamic`, custom `trtexec` flags):**

---

This is often not a bug — see the README section "What to do with the results."
Pure INT8 slower than FP16 is a common outcome on small or attention-heavy
models. Still, paste the report and I'll take a look.
