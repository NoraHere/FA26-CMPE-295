# Factual-knowledge judges under quantization

Part of the Q-Judge project: do quantized small LLM judges stay accurate and hard to fool?
The team's main experiments use GSM8K math. This folder tests the same question on **factual
knowledge** and compares more quantization methods.

Each judge answers Yes/No (A/B for pairs). The score is the first-token margin
logit(Yes) − logit(No), with thinking off and no sampling.

## Setup

**Judges:** Qwen3-4B, Qwen3-8B, Qwen2.5-7B-Instruct, Phi-4-mini-instruct

**Datasets**

| Dataset | Size | What is measured |
|---|---|---|
| TruthfulQA | 5,015 answers | Yes/No per answer, graded without a key (`noref`) and with `best_answer` as key (`ref`) |
| RewardBench 2 Factuality (RB2) | 475 prompts × 4 answers (1 correct) | correct-is-top (correct answer scored highest, chance 25%) and % of wrong answers accepted |
| JudgeBench knowledge | 154 MMLU-Pro pairs (GPT split) | pairwise accuracy, judged in both orders |

**Quantization methods**

| Name | Team name | Calibration | Real / simulated |
|---|---|---|---|
| BF16 | bf16 | – | reference |
| INT8 (torchao, weight-only) | tao_w8 | no | real |
| INT4 (torchao, weight-only) | tao_w4 | no | real |
| RTN4 | rtn_w4 | no | simulated (team `simquant.py`) |
| GPTQ4 | gptq_w4 | yes | simulated (team `simquant.py`) |
| MNR4 (GPTQ + margin-neutral rounding) | mnr_w4 | yes | simulated (team `neutral.py`) |
| GPTAQ4 (asymmetric calibration, `gptaq.py`) | – | yes | simulated |
| GPTQ / AWQ W4A16 | – | yes | real checkpoints (llm-compressor 0.8.1) |
| HQQ 4-bit | – | no | real |
| FP8 W8A8, INT8 W8A8 | – | no | real, H100 only |

Simulated methods replace each linear layer with its dequantized 4-bit weights and run in BF16,
so they measure accuracy, not memory or speed. All weights use group size 128; embeddings and
`lm_head` are not quantized.

Calibration (GPTQ, AWQ, MNR, GPTAQ): 128 judge-format prompts from RB2 subsets other than
Factuality (seed 0), so there is no overlap with any scored item. One calibration set was used.

GPTAQ: GPTQ is run on the target W* = W + α·W(X−X̃)ᵀX̃(H + c·mean(diag H)·I)⁻¹, with H = X̃ᵀX̃,
X the full-precision input and X̃ the quantized-path input. Settings α = 0.25, c = 1.0 were chosen
from the size of the correction only (α = 1 with 1% damping made the correction blow up).

## Results

"Agree" = share of verdicts that match the same judge in BF16. Full output with bootstrap
CIs: `results/all_results.txt`.

### RB2 Factuality: correct-is-top / wrong accepted / agree (CI about ±4 points)

| Method | Qwen3-4B | Qwen3-8B | Qwen2.5-7B | Phi-4-mini |
|---|---|---|---|---|
| BF16 | 72.0 / 28.2 / 100 | 78.3 / 27.2 / 100 | 60.0 / 45.6 / 100 | 36.4 / 94.5 / 100 |
| INT8 | 71.8 / 28.4 / 98.1 | 78.1 / 29.6 / 98.1 | 60.6 / 44.3 / 96.8 | 35.4 / 94.3 / 89.3 |
| torchao INT4 | 66.5 / **57.1** / 81.9 | 72.2 / **53.5** / 86.3 | 59.6 / 43.7 / 83.6 | 33.7 / 96.1 / 78.7 |
| RTN4 | 70.3 / 16.7 / 86.9 | 77.5 / 31.2 / 90.3 | 59.6 / 45.0 / 83.2 | 38.3 / 93.8 / 74.9 |
| GPTQ4 | 71.6 / 21.5 / 88.6 | 76.8 / 28.5 / 93.1 | 60.2 / 43.7 / 86.7 | 39.2 / 93.8 / 85.9 |
| GPTQ (real) | 68.8 / 19.2 / 88.4 | 76.6 / 23.1 / 92.0 | 64.0 / 42.9 / 87.6 | 35.6 / 93.7 / 86.5 |
| AWQ (real) | 71.2 / **49.5** / 86.1 | 76.8 / 30.4 / 88.8 | 58.1 / 41.8 / 83.8 | 37.7 / 96.3 / 78.5 |
| HQQ | 71.2 / 16.9 / 86.5 | 76.0 / 34.9 / 88.4 | 58.5 / **54.9** / 83.8 | 37.7 / 96.1 / 74.3 |
| MNR4 | 71.4 / 23.0 / 89.3 | 77.3 / 26.5 / 90.9 | 59.2 / 41.7 / 86.5 | 40.8 / 93.8 / 84.2 |
| GPTAQ4 | 72.0 / 21.5 / 90.3 | 79.8 / 26.3 / 93.5 | 58.5 / 44.8 / 88.4 | 39.4 / 94.2 / 87.4 |

### TruthfulQA without key: AUROC / wrong accepted / agree

| Method | Qwen3-4B | Qwen3-8B | Qwen2.5-7B | Phi-4-mini |
|---|---|---|---|---|
| BF16 | 0.770 / 6.9 / 100 | 0.797 / 8.5 / 100 | 0.742 / 7.7 / 100 | 0.714 / 49.4 / 100 |
| INT8 | 0.769 / 6.5 / 99.0 | 0.800 / 9.0 / 98.8 | 0.742 / 7.7 / 99.3 | 0.713 / 50.6 / 98.2 |
| torchao INT4 | 0.758 / 13.1 / 88.2 | 0.785 / 11.2 / 92.2 | 0.761 / 4.5 / 93.7 | 0.697 / 53.3 / 89.0 |
| RTN4 | 0.768 / 5.9 / 93.6 | 0.781 / 10.4 / 93.6 | 0.746 / 8.6 / 95.0 | 0.660 / 46.4 / 87.7 |
| GPTQ4 | 0.782 / 6.1 / 95.8 | 0.785 / 8.8 / 96.0 | 0.732 / 8.3 / 95.9 | 0.701 / 51.0 / 93.3 |
| GPTQ (real) | 0.767 / 7.1 / 95.3 | 0.786 / 8.3 / 95.2 | 0.733 / 7.4 / 94.9 | 0.702 / 50.9 / 93.0 |
| AWQ (real) | 0.755 / 8.2 / 93.2 | 0.766 / 9.2 / 93.8 | 0.715 / 7.2 / 93.6 | 0.710 / 56.1 / 89.6 |
| HQQ | 0.767 / 4.2 / 94.0 | 0.783 / 10.3 / 93.0 | 0.733 / 8.9 / 95.2 | **0.626 / 66.5** / 83.8 |
| MNR4 | 0.779 / 5.5 / 95.6 | 0.785 / 8.8 / 95.4 | 0.736 / 7.3 / 95.8 | 0.716 / 45.1 / 92.6 |
| GPTAQ4 | 0.771 / 6.2 / 95.9 | 0.792 / 7.6 / 96.0 | 0.735 / 7.7 / 96.1 | 0.706 / 50.2 / 93.9 |

With the key, all versions reach AUROC 0.85–0.93 and agreement stays at 87–99.6%.

### JudgeBench knowledge: accuracy / agree (CI about ±8 points)

| Method | Qwen3-4B | Qwen3-8B | Qwen2.5-7B | Phi-4-mini |
|---|---|---|---|---|
| BF16 | 59.1 / 100 | 61.0 / 100 | 58.4 / 100 | 53.9 / 100 |
| INT8 | 60.4 / 95.5 | 59.1 / 99.7 | 57.8 / 98.4 | 53.9 / 95.5 |
| torchao INT4 | 57.1 / 80.2 | 63.6 / 93.8 | 55.8 / 79.9 | 51.9 / 76.0 |
| RTN4 | 61.0 / 79.2 | 61.7 / 91.9 | 57.1 / 86.0 | 59.7 / 76.0 |
| GPTQ4 | 60.4 / 89.3 | 60.4 / 95.8 | 59.7 / 94.5 | 50.6 / 78.2 |
| MNR4 | 59.1 / 84.1 | 63.0 / 97.1 | 57.8 / 92.2 | 51.9 / 94.2 |
| GPTAQ4 | 63.0 / 89.9 | 65.6 / 96.4 | 59.7 / 89.0 | 51.9 / 90.3 |

Every version is near chance, so this set mostly shows position bias (Qwen3-8B prefers the first
answer in 91–99% of pairs).

### Average over the 4 judges

| Method | Agree TQA noref / ref / RB2 / JB | Mean \|Δ RB2 wrong accepted\| | Worst |
|---|---|---|---|
| INT8 | 98.8 / 99.4 / 95.6 / 97.3 | 1.0 | 2.4 |
| GPTAQ4 | 95.5 / 97.8 / 89.9 / 91.4 | 2.2 | 6.7 |
| MNR4 | 94.8 / 97.4 / 87.7 / 91.9 | 2.6 | 5.2 |
| GPTQ4 | 95.3 / 97.6 / 88.6 / 89.5 | 2.7 | 6.7 |
| GPTQ (real) | 94.6 / 97.3 / 88.6 / – | 4.1 | 9.0 |
| RTN4 | 92.5 / 96.1 / 83.8 / 83.3 | 4.2 | 11.5 |
| AWQ (real) | 92.6 / 95.2 / 84.3 / – | 7.5 | 21.3 |
| HQQ | 91.5 / 94.5 / 83.2 / – | 7.5 | 11.3 |
| torchao INT4 | 90.8 / 94.9 / 82.6 / 82.5 | 14.7 | 28.9 |

### Memory and speed (RB2, A100): GiB / items per second

| Method | Qwen3-4B | Qwen3-8B | Qwen2.5-7B | Phi-4-mini |
|---|---|---|---|---|
| BF16 | 7.5 / 108 | 15.3 / 68 | 14.2 / 83 | 7.1 / 149 |
| INT8 | 4.1 / 97 | 8.8 / 61 | 8.1 / 73 | 4.2 / 135 |
| torchao INT4 | 2.8 / 14 | 5.8 / 8 | 5.6 / 8 | 2.7 / 20 |
| GPTQ (real) | 3.2 / 91 | 5.7 / 57 | 5.2 / 68 | 3.8 / 124 |
| AWQ (real) | 3.2 / 90 | 5.7 / 57 | 5.2 / 67 | 3.8 / 122 |
| HQQ | 2.5 / 98 | 5.8 / 61 | 5.3 / 72 | 2.7 / 131 |

On H100, FP8 W8A8 (Qwen3-8B) matches INT8 accuracy (RB2 77.7 vs 78.3 BF16) but gives no speed-up
in plain Transformers (119 vs 128 items/s). INT8 W8A8 is less accurate (74.1) and 5× slower.

## Findings

1. **INT8 is safe**: 96–99.7% agreement with BF16 on every dataset, about 40% less memory.
2. **4-bit is safe only with GPTQ-style calibration.** GPTQ, MNR and GPTAQ stay close to BF16 for
   all judges and never make a judge easier to fool. The real llm-compressor GPTQ checkpoint
   matches the simulated one and runs at about 84% of BF16 speed.
3. **4-bit without calibration drifts unpredictably.** torchao INT4 doubles the wrong answers
   Qwen3 accepts (28% → 57%) and is 7–8× slower. RTN and HQQ make Qwen3-4B stricter instead; HQQ
   makes Qwen2.5-7B more lenient and hurts Phi-4-mini. AWQ makes Qwen3-4B lenient (49.5%).
4. **GPTAQ is the steadiest 4-bit version** (highest agreement in 13 of 16 judge × dataset cells),
   but its accuracy differences from GPTQ are within the CIs.
5. **Judge choice matters more than quantization.** Phi-4-mini accepts about 95% of wrong RB2
   answers even in BF16; with a reference answer it matches Qwen3 (AUROC 0.926). Qwen3-8B is the
   best fact judge.

**Limitations:** one calibration seed; RB2 CIs about ±4 points and JudgeBench about ±8; MNR is
designed against fake "verified" notes, which none of these datasets contain.

## Files

```
scripts/
  fact_eval.py          scores TruthfulQA, RB2 and JudgeBench knowledge for one judge/version
  judge_eval.py         Judge class: model loading, quantization configs, margins
  gptaq.py              GPTAQ-style calibration
  llmc_quant.py         real GPTQ / AWQ W4A16 checkpoints with llm-compressor
  analyze_fact.py       tables with bootstrap CIs -> results/fact_summary.json
  download.sh, download_judges.sh   models and datasets from Hugging Face
  pick_gpu.sh           picks a working GPU from the allocation
  run_*.sbatch          Slurm job arrays for each batch of runs
results/
  all_results.txt       full table (output of analyze_fact.py)
  fact_summary.json
  runs/<judge>-<version>/   per-item scores (truthfulqa.csv, rb2_factuality.csv,
                            judgebench_knowledge.csv) and run metadata
```

## Running

The scripts expect the working directory `~/qjudge` (scripts in `~/qjudge/scripts`, models in
`~/qjudge/models`, data in `~/qjudge/data`). The simulated methods (`rtn4`, `gptq4`, `mnr4`,
`gptaq4`) import the team repo from `~/q-judge`.

Environment on the SJSU HPC: `module load python3/3.12.12 ml/lambda/gpu-ai ml/torch/2.6`
(torch 2.6, transformers 4.57.3, torchao 0.9.0, compressed-tensors 0.12.2). llm-compressor
0.8.1 and hqq 0.2.8.post1 are installed `--no-deps` into separate directories added to
`PYTHONPATH` only for those runs.

```bash
# one judge / version
python scripts/fact_eval.py --model ~/qjudge/models/Qwen3-8B --quant gptq4
# --quant: bf16, int8, int4, w8a8, fp8, rtn4, gptq4, mnr4, gptaq4, hqq4
# --only jb   JudgeBench knowledge only

# Slurm (submit from ~/qjudge so logs/ resolves)
sbatch scripts/run_fact.sbatch

# tables
python scripts/analyze_fact.py
```

Notes:
- MNR needs about 3× the BF16 weights in GPU memory; run the 7–8B judges on an H100.
- transformers 4.57.3 always loads HQQ models in float32; `judge_eval.py` patches the quantizer to bf16.
- llm-compressor's save step imports DeepSpeed, which fails without `CUDA_HOME`; `llmc_quant.py` disables that check.
