# judgebench/ (Jiayang Wang)

JudgeBench (GPT split) math + code evaluation of small LLM judges, using the same
first-token logit protocol as the team qjudge harness. Standalone for now; meant to
become a qjudge stage once that repo lands.

## Layout

| path | what |
|---|---|
| `jb_eval.py` | evaluator: `--mode pointwise | pairwise | both`, `--limit N` for a stratified subsample |
| `jobs/setup_env.sbatch` | one-time offline install of the pinned stack from `~/wheelhouse` into the GPU venv |
| `jobs/jb_smoke.sbatch` | 4 pairs (2 math + 2 code), both modes, Qwen3-4B, plus a forced-truncation run at max-len 1024 |
| `jobs/jb_full.sbatch` | all 109 pairs, both modes; `MODEL=... DTYPE=...` overrides |
| `results/<run>/` | `items.jsonl` (one line per pair x mode), `meta.json`, `summary.json`, `summary.md` |
| `docs/jb_mathcode_pair_ids.tsv` | the 109 pair ids (56 livebench-math, 11 mmlu-pro-math, 42 livecodebench) |
| `notebooks/` | earlier generation-based A/B pilot (superseded by `jb_eval.py`) |

## Running

GPU work only through Slurm on coe-hpc3 (partition `gpuqs`, A100; P100 cannot run INT4).
The venv's Python 3.11 exists only on the compute nodes.

```bash
ssh coe-hpc3
cd ~/judgebench
J=$(sbatch --parsable jobs/setup_env.sbatch)   # once
sbatch --dependency=afterok:$J jobs/jb_smoke.sbatch   # then: tail logs/jb_smoke-<jobid>.out
sbatch jobs/jb_full.sbatch
```

## Method

- **pointwise**: prompt asks CORRECT/INCORRECT for one response; margin =
  logit(CORRECT) - logit(INCORRECT) on the first generated token. Higher margin wins; exact tie = wrong.
- **pairwise**: prompt asks A or B; m = logit(A) - logit(B), run in original and swapped order.
  debiased = (m_orig - m_rev) / 2. Reports accuracy (orig / rev / debiased), order consistency,
  first-slot pick rate.
- Thinking off (`enable_thinking=False`), no sampling, single forward pass.
- Label tokens are picked by probability mass on two fixed calibration prompts (not JudgeBench,
  not GSM8K); the choice and mass are stored in `meta.json`.
- Over-long prompts truncate the responses (longest first), never the question or instruction tail.
- 95% percentile bootstrap, 2000 resamples, clustered by `original_id`. Note: `original_id` is null
  for all livebench-math and livecodebench pairs (98 of 109) and every question is unique, so those
  pairs fall back to `pair_id` as their own cluster.
- Gold labels are read only after the model call, for scoring.

## Not yet done

- Quantized conditions (INT8 / INT4 RTN / INT4 GPTQ): waiting on qjudge loaders.
- Pointwise prompt should be swapped for the exact qjudge prompt when available
  (`POINTWISE_TEMPLATE`; its hash is in `meta.json` so runs stay distinguishable).
- Team judges downloaded: Qwen3-1.7B, Qwen3-4B. Still to fetch: Qwen3-8B, Phi-4-mini, Granite-3.3-8B.
