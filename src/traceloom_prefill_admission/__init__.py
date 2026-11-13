"""Reserve prompt capacity for short requests while keeping the total step budget. Uses stock vLLM settings; no runtime patch."""
from __future__ import annotations
from copy import deepcopy
import argparse
import json
from pathlib import Path

__version__ = "0.1.0"

def _integer(value, expected, name):
    if type(value) is not int or value not in expected:
        raise ValueError(f"{name} must be one of {sorted(expected)}")

def configure(engine_kwargs, *, enabled=False):
    """Return independent LLM kwargs. Installation and the default are inert."""
    result = deepcopy(engine_kwargs)
    if not enabled:
        return result
    _integer(result.get("max_num_seqs"), {4}, "max_num_seqs")
    if result.get("enforce_eager", False):
        raise ValueError("the qualified recipe requires graph execution")
    if result.get("enable_chunked_prefill") is not True:
        raise ValueError("enable_chunked_prefill=True is required")
    if result.get("enable_prefix_caching") is not False:
        raise ValueError("enable_prefix_caching=False is required")
    _integer(result.get("max_num_batched_tokens"), {256}, "max_num_batched_tokens")
    _integer(result.get("long_prefill_token_threshold", 0), {0, 224}, "long_prefill_token_threshold")
    if result.get("max_model_len", 0) < 2080:
        raise ValueError("max_model_len must cover the qualified 2048+32-token request")
    expected = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    compilation = result.get("compilation_config", {})
    if not isinstance(compilation, dict) or compilation.get("cudagraph_capture_sizes") != expected:
        raise ValueError("retain the qualified capture sizes [1,2,4,...,256]")
    result["long_prefill_token_threshold"] = 224
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="JSON object containing existing LLM kwargs")
    p.add_argument("--enable", action="store_true", help="explicitly apply this recipe")
    a = p.parse_args()
    print(json.dumps(configure(json.loads(a.input.read_text()), enabled=a.enable), indent=2))
