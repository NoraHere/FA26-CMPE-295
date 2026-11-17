"""Real 4-bit judge checkpoints with llm-compressor (W4A16, group 128, lm_head kept BF16).

  --method gptq : GPTQModifier, scheme W4A16 (same scheme as RedHatAI/Qwen3-4B-quantized.w4a16)
  --method awq  : AWQModifier,  scheme W4A16

Calibration = the same 128 judge-format prompts fact_eval.py --quant gptq4 uses (RB2 subsets
other than Factuality, rendered with the fact templates + the judge's chat template), so the
real-kernel and simulated GPTQ runs see identical calibration data.

Output: ~/qjudge/models/<Model>-llmc-<method>-w4a16 (compressed-tensors format). Score it with
  fact_eval.py --model <that dir> --quant bf16   (loads through compressed-tensors' own config)

Runs in a separate lib dir (~/qjudge/pylib_llmc, llmcompressor 0.8.1 installed --no-deps):
its declared pins (torch>=2.7, transformers<=4.56.2) would replace the eval stack.
"""
import argparse, json, os, sys, time
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fact_eval import calibration_prompts


class _J:  # the bits of judge_eval.Judge that calibration_prompts needs
    def __init__(self, tok):
        self.tok = tok

    def trunc(self, text, n):
        ids = self.tok.encode(text, add_special_tokens=False)
        return text if len(ids) <= n else self.tok.decode(ids[:n]) + "\n...[truncated]"

    def render(self, sys_, user):
        return self.tok.apply_chat_template(
            [{"role": "system", "content": sys_}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", choices=["gptq", "awq"], required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    src = os.path.expanduser(args.model.rstrip("/"))
    out = f"{src}-llmc-{args.method}-w4a16"
    if os.path.exists(f"{out}/config.json") and not args.overwrite:
        print(f"[llmc] {out} exists; --overwrite to rebuild"); return

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.modifiers.awq import AWQModifier
    import llmcompressor, compressed_tensors

    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16, device_map="cuda")
    calib, subsets = calibration_prompts(_J(tok))
    ds = Dataset.from_dict({"input_ids": [tok.encode(c, add_special_tokens=False) for c in calib]})
    ntok = sum(len(x) for x in ds["input_ids"])
    print(f"[llmc] {len(calib)} calibration prompts, {ntok} tokens", flush=True)

    torch.manual_seed(1234)
    if args.method == "gptq":
        recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"], dampening_frac=0.01)
    else:
        recipe = AWQModifier(targets=["Linear"], scheme="W4A16", ignore=["lm_head"])
    t0 = time.time()
    oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=1024,
            num_calibration_samples=len(calib))
    dt = time.time() - t0
    # accelerate's unwrap_model imports deepspeed when it is installed (it is, in ml/torch/2.6), and
    # deepspeed raises MissingCUDAException without CUDA_HOME. Nothing here uses deepspeed.
    import accelerate.utils.other as _acc_other
    _acc_other.is_deepspeed_available = lambda: False
    model.save_pretrained(out, save_compressed=True)
    tok.save_pretrained(out)
    json.dump(dict(source=src, method=args.method, scheme="W4A16", group_size=128, ignore=["lm_head"],
                   calib_n=len(calib), calib_tokens=ntok, calib_subsets=subsets, quantize_s=round(dt, 1),
                   llmcompressor=llmcompressor.__version__, compressed_tensors=compressed_tensors.__version__,
                   torch=torch.__version__),
              open(f"{out}/qjudge_llmc_manifest.json", "w"), indent=1)
    print(f"[llmc] {args.method} done in {dt:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
