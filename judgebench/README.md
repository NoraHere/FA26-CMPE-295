# judgebench/ (Jiayang Wang)

JudgeBench (GPT split) evaluation of small LLM judges (math + code by default; knowledge and
reasoning with `--tasks`), using the same
first-token logit protocol as the team qjudge harness. Standalone for now; meant to
become a qjudge stage once that repo lands.

## Layout

| path | what |
|---|---|
| `jb_eval.py` | evaluator: `--mode pointwise | pairwise | both`, `--tasks math,code,reasoning,knowledge`, `--limit N` for a subsample stratified by task, `--grid` + `--method rtn|gptq|mnr` for a quantized judge |
| `quant_grids.py` | offset-shifted round-to-nearest weight grids (fake quant), shared by `jb_eval.py` and S9: `bf16`, INT4 `g00..g15`, INT8 `i8g00..i8g15` |
| `quant_methods.py` | rounding on a fixed grid: own GPTQ (wikitext-2 calibration) and margin-neutral rounding (MNR: GPTQ plus a few flips that cancel the note-effect change vs BF16, tuned on S9 development problems); `python quant_methods.py` self-test |
| `jb_compare.py` | grid runs vs the BF16 run: accuracy, paired change vs BF16 (bootstrap CI), agreement with BF16 |
| `s9_build_injection.py` | S9 data (CPU): GSM8K wrong answers with/without "verified" notes, hash partitions, gold kept in `audit.jsonl` |
| `s9_score_grids.py` | S9 scoring (GPU): BF16 + 16 offset-shifted INT4 grids (same scale and worst-case error) + INT8 control grids |
| `s9_lottery_stats.py` | S9 rounding-lottery statistics (CPU): spread, split-half, transfer, audit; `--synthetic` self-test |
| `jobs/s9_grids.sbatch` | S9 build + score + stats (16 INT4 + 16 INT8 grids, primary note `strong`); smoke via `PARTITIONS=discovery LIMIT=4 GRIDS=2 INT8=2 EVAL=all` |
| `docs/s9_preregistration.md` | S9 hypothesis and pass/fail rule (draft; freeze before development/locked runs) |
| `jobs/setup_env.sbatch` | one-time offline install of the pinned stack (plus the IFEval checker packages) from `~/wheelhouse` into the GPU venv |
| `jobs/jb_smoke.sbatch` | 4 pairs (2 math + 2 code), both modes, Qwen3-4B, plus a forced-truncation run at max-len 1024 |
| `jobs/jb_full.sbatch` | all 109 pairs, both modes, Qwen3-4B BF16; `MODEL=... DTYPE=...` overrides |
| `jobs/jb_methods_smoke.sbatch` | smoke test of knowledge + reasoning and of gptq / mnr (4 pairs, small calibration), plus S9 mnr scoring on 2 grids |
| `jobs/jb_methods.sbatch` | one judge on `TASKS` (default reasoning): BF16, INT8 RTN, INT4 RTN / GPTQ / MNR on grid `GRID` (default 00), H100 |
| `jobs/jb_grids.sbatch` | all 109 pairs on each grid (default the 16 INT4 grids; `GRIDS="i8g00 ..."` overrides) |
| `results/<run>/` | `items.jsonl` (one line per pair x mode), `meta.json`, `summary.json`, `summary.md` |
| `docs/jb_mathcode_pair_ids.tsv` | the 109 pair ids (56 livebench-math, 11 mmlu-pro-math, 42 livecodebench) |
| `notebooks/` | earlier generation-based A/B pilot (superseded by `jb_eval.py`) |

## Running

GPU work only through Slurm on coe-hpc3 (partition `gpuqs`, A100 or `--gres=gpu:h100:1`; P100 cannot
run INT4). Some A100s fail with "No CUDA GPUs are available" (cs002 GPU 0; cs001 on 2026-10-31); use H100 then.
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

- Quantized conditions: own offset-shifted RTN grids, own GPTQ and own margin-neutral rounding (all fake
  quant, `--grid` + `--method`) are available; torchao INT8/INT4 still to add for comparability with the team.
- Pointwise prompt should be swapped for the exact qjudge prompt when available
  (`POINTWISE_TEMPLATE`; its hash is in `meta.json` so runs stay distinguishable).
- Team judges downloaded: Qwen3-1.7B, Qwen3-4B, Qwen3-8B, Phi-4-mini-instruct, Granite-3.3-8B-Instruct.
  Granite's pointwise label pair is 'CORRECT' / 'IN' (first tokens), mass 0.76 on calibration.
- Pairwise mode: Qwen3-8B (and Qwen3-4B on math) usually starts with "To" / "The" instead of A/B
  (label mass < 0.1), so pairwise margins there are unreliable; pointwise is the primary protocol.
