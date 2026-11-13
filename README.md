# TraceLoom Prefill Admission

Set the stock long_prefill_token_threshold to 224 while retaining the global 256-token step budget, to let short requests share a prompt step.

This is an independently installable **configuration MOD**, implemented with
existing vLLM controls. Installing it changes no engine behavior. TraceLoom
supported the diagnosis and selection of the setting; the package adds a
checked, explicit application of that setting.

## Install and use

```bash
git clone https://github.com/vLLM-HUST/vllm-hust-prefill-admission.git
cd vllm-hust-prefill-admission
python -m pip install .
traceloom-prefill-admission examples/base.json --enable
```

The command prints LLM kwargs; it does not start an engine. In an application:

```python
import json
from traceloom_prefill_admission import configure
from vllm import LLM

base = json.load(open("examples/base.json"))
base["model"] = "/path/to/Qwen3-0.6B"
llm = LLM(**configure(base, enabled=True))
```

`configure(base)` returns an unchanged independent copy. Unsupported qualified
contexts and conflicting settings raise an error before engine construction.
The original dict remains available for rollback. After obtaining a device
allocation, `examples/run.py --model PATH --enable` runs an explicit smoke
example; it is not the historical performance measurement.

## Result and compatibility

Short first-token latency changed from 126.746 to 40.342 ms (68.17% lower); batch completion changed from 505.131 to 515.347 ms (2.02% higher).

One 2048-token prompt arrives at 0 ms; three 8-token prompts at 50 ms; all request 32 output tokens. Qwen3-0.6B FP16, TP1, Ascend910B2; five independent process pairs and 60 unprofiled batches. Corresponding output tokens matched.

Historical engine revisions: vLLM `0fc695f` and vLLM-Ascend `f4a08bd`,
TorchNPU `2.10.0.post4`. The setting API is checked by this package;
performance on a different engine revision requires a new paired run.
See [evidence](docs/EVIDENCE.md), [protocol](evidence/protocol.json), and
[provenance](docs/PROVENANCE.md).

This is a latency/throughput tradeoff: batch completion increased by 2.02%. The reserved capacity is workload-specific. Keep the capture sizes unchanged and compare short-request TTFT with total batch completion.

## Disable and verify

Restart with the original kwargs (`enabled=False`), then optionally run
`python -m pip uninstall vllm-hust-prefill-admission`. No source files are patched.

```bash
python -m pip install '.[test]'
python -m pytest -q
python -m build
```

Maintainer: [Shuhao Zhang](https://github.com/ShuhaoZhangTony).
