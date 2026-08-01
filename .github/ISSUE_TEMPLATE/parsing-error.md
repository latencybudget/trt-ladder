---
name: trtexec output failed to parse
about: The tool errored with "no performance summary in trtexec output"
title: "[parse] "
labels: parser
---

**TensorRT version** (`trtexec --help` first line):

**Full trtexec log**, if you can share it (the `.log` file next to the `.engine`
in your output directory):

```
paste here
```

trtexec's output format has changed across major versions before. If yours
looks different from what's parsed in `trt_ladder/trtexec.py`, pasting the log
is the fastest way to get it fixed.
