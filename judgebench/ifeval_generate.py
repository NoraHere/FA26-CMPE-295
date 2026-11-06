"""IFEval step 1 (GPU): generate responses to the 541 IFEval prompts, then label them with the
official IFEval rule checkers (third_party/instruction_following_eval).

Generators are the team's: Qwen3-1.7B (thinking off) and SmolLM2-1.7B-Instruct, --samples responses
each per prompt, sampled (temperature 0.7, top-p 0.95, fixed seed) so that some responses follow the
rules and some do not. Generators are never judges, and the labels never enter a judge prompt.

Output: data/ifeval/responses.jsonl, one row per response:
  key, generator, sample, response, new_tokens, finished (ended with EOS, not cut at --max-new-tokens),
  follow_strict (all rules followed, official strict check), follow_list_strict, follow_loose
plus data/ifeval/generate_meta.json. Resumable per (generator, prompt).

Usage (GPU node only):
  python ifeval_generate.py --limit 8      # smoke
  python ifeval_generate.py
  python ifeval_generate.py --label-only   # re-run the checkers on existing responses (CPU is fine)
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "third_party"))

DEFAULT_PROMPTS = "/home/019111677/datasets/IFEval/ifeval_input_data.jsonl"
GENERATORS = {
    "qwen3-1.7b": "/home/019111677/models/Qwen3-1.7B",
    "smollm2-1.7b": "/home/019111677/models/SmolLM2-1.7B-Instruct",
}


def load_prompts(path, limit):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows.sort(key=lambda r: r["key"])
    return rows[:limit] if limit else rows


def label(rows, prompts):
    """Add official strict and loose IFEval labels to each response row (in place)."""
    from instruction_following_eval import evaluation_lib as ev
    by_key = {p["key"]: p for p in prompts}
    for r in rows:
        p = by_key[r["key"]]
        inp = ev.InputExample(key=p["key"], instruction_id_list=p["instruction_id_list"], prompt=p["prompt"],
                              kwargs=[{k: v for k, v in kw.items() if v is not None} for kw in p["kwargs"]])
        strict = ev.test_instruction_following_strict(inp, {p["prompt"]: r["response"]})
        loose = ev.test_instruction_following_loose(inp, {p["prompt"]: r["response"]})
        r["follow_strict"] = strict.follow_all_instructions
        r["follow_list_strict"] = strict.follow_instruction_list
        r["follow_loose"] = loose.follow_all_instructions
    return rows


def generate(name, path, prompts, args, out_path, done):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={"": 0},
                                                 local_files_only=True).eval()
    todo = [p for p in prompts if (name, p["key"]) not in done]
    print(f"[{name}] {len(todo)} prompts x {args.samples} samples", flush=True)
    stop = [tok.eos_token_id] + ([tok.convert_tokens_to_ids("<|im_end|>")] if "<|im_end|>" in tok.get_vocab() else [])
    t0 = time.time()
    for k in range(0, len(todo), args.batch_prompts):
        batch = todo[k:k + args.batch_prompts]
        chats = [tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False) for p in batch]
        enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(0)
        torch.manual_seed(args.seed + k)
        with torch.inference_mode():
            gen = model.generate(**enc, do_sample=True, temperature=0.7, top_p=0.95,
                                 max_new_tokens=args.max_new_tokens, num_return_sequences=args.samples,
                                 eos_token_id=stop, pad_token_id=tok.pad_token_id)
        new = gen[:, enc.input_ids.shape[1]:]
        rows = []
        for i, p in enumerate(batch):
            for s in range(args.samples):
                ids = new[i * args.samples + s].tolist()
                finished = any(t in stop for t in ids)
                ids = [t for t in ids if t not in stop and t != tok.pad_token_id]
                rows.append({"key": p["key"], "generator": name, "sample": s,
                             "response": tok.decode(ids, skip_special_tokens=True).strip(),
                             "new_tokens": len(ids), "finished": finished})
        label(rows, prompts)
        with out_path.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n = k + len(batch)
        print(f"[{name}] {n}/{len(todo)} prompts, {time.time() - t0:.0f}s; batch strict pass "
              f"{sum(r['follow_strict'] for r in rows)}/{len(rows)}", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", default=DEFAULT_PROMPTS)
    ap.add_argument("--out-dir", default=str(HERE / "data" / "ifeval"))
    ap.add_argument("--generators", default=",".join(GENERATORS))
    ap.add_argument("--samples", type=int, default=4, help="responses per prompt per generator")
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--batch-prompts", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="first N prompts (by key); 0 = all 541")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label-only", action="store_true")
    args = ap.parse_args()

    prompts = load_prompts(args.prompts, args.limit)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "responses.jsonl"
    rows = []
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]

    if args.label_only:
        label(rows, prompts)
        out_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    else:
        done = {(r["generator"], r["key"]) for r in rows}
        for name in args.generators.split(","):
            generate(name, GENERATORS[name], prompts, args, out_path, done)
        with out_path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]

    keys = {p["key"] for p in prompts}
    rows = [r for r in rows if r["key"] in keys]
    stats = {}
    for name in sorted({r["generator"] for r in rows}):
        sub = [r for r in rows if r["generator"] == name]
        fin = [r for r in sub if r["finished"]]
        stats[name] = {"responses": len(sub), "finished": len(fin),
                       "strict_pass": sum(r["follow_strict"] for r in fin),
                       "loose_pass": sum(r["follow_loose"] for r in fin),
                       "prompts_with_pass_and_fail": len(
                           {r["key"] for r in fin if r["follow_strict"]} & {r["key"] for r in fin if not r["follow_strict"]})}
    both = [r for r in rows if r["finished"]]
    stats["all"] = {"prompts_with_pass_and_fail": len(
        {r["key"] for r in both if r["follow_strict"]} & {r["key"] for r in both if not r["follow_strict"]})}
    meta = {"prompts": args.prompts, "n_prompts": len(prompts), "generators": args.generators,
            "samples": args.samples, "max_new_tokens": args.max_new_tokens, "seed": args.seed,
            "sampling": {"temperature": 0.7, "top_p": 0.95}, "stats": stats}
    (out_dir / "generate_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
