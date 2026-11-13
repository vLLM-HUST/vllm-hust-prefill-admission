import json
from pathlib import Path
import pytest
from traceloom_prefill_admission import configure

def base():
    return json.loads((Path(__file__).parents[1]/"examples/base.json").read_text())

def test_disabled_is_inert_and_copy_is_independent():
    x=base();y=configure(x)
    assert y==x and y is not x
    y.setdefault("compilation_config",{})["test"]=1
    assert "test" not in x.get("compilation_config",{})

@pytest.mark.parametrize("field,value",[("max_num_seqs",8),("enforce_eager",True),("enable_chunked_prefill",False),("enable_prefix_caching",True)])
def test_incompatible_mode(field,value):
    x=base();x[field]=value
    with pytest.raises(ValueError):configure(x,enabled=True)

def test_enabled_changes_cap_only():
    x=base(); y=configure(x,enabled=True)
    assert y.pop("long_prefill_token_threshold")==224
    x.pop("long_prefill_token_threshold")
    assert y==x
    assert y["max_num_batched_tokens"]==256

def test_wrong_budget_and_capture_refused():
    x=base();x["max_num_batched_tokens"]=224
    with pytest.raises(ValueError):configure(x,enabled=True)
    x=base();x["compilation_config"]["cudagraph_capture_sizes"]=[1,2,4]
    with pytest.raises(ValueError):configure(x,enabled=True)

