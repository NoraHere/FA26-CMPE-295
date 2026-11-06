# third_party

`instruction_following_eval/`: IFEval rule checkers, copied unchanged from
google-research/google-research (`instruction_following_eval/`, master, fetched 2026-11-04), Apache 2.0
(see its LICENSE). Used only to label generated IFEval responses; the judge never sees these labels.
Needs `langdetect nltk immutabledict absl-py` (in `jobs/setup_env.sbatch`) and nltk `punkt`/`punkt_tab`
data in `~/nltk_data`.
