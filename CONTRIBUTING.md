# Contributing

Issues and PRs are welcome, especially:

- trtexec output from a TensorRT version whose log format the parser
  (`trt_ladder/trtexec.py`) doesn't handle correctly
- a model/precision combination that gave a surprising or wrong-looking result
- JetPack/device combinations not yet covered by `trt_ladder/env.py`'s
  clock-detection glob

For a bug report, the most useful thing you can attach is the raw `.log` file
`trt-ladder` writes next to each `.engine` — the parser is regex-based against
real trtexec output, so a mismatched log is usually a five-minute fix once
someone can see it.

Run the test suite before submitting a PR:

```bash
python tests/test_parse_and_report.py
```
