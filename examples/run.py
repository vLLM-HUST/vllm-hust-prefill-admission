"""Explicit user launch; acquire your device allocation before running."""
import argparse, json
from pathlib import Path
from traceloom_prefill_admission import configure

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--enable", action="store_true")
    a = p.parse_args()
    kwargs = json.loads(Path(__file__).with_name("base.json").read_text())
    kwargs["model"] = a.model
    effective = configure(kwargs, enabled=a.enable)
    print(json.dumps(effective, indent=2), flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(**effective)
    for output in llm.generate(["Explain graph execution."], SamplingParams(temperature=0, max_tokens=16)):
        print(output.outputs[0].text)

if __name__ == "__main__":
    main()
