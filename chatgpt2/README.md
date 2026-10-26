# ChatGPT Pro rounds

The briefs sent to ChatGPT Pro are in `questions/`. Each is kept as sent, so some of their numbers predate later corrections; `../REPORT.md` is authoritative. The answer files are numbered separately from the briefs.

| brief | answer | code and results | checked in REPORT |
|---|---|---|---|
| `questions/QUESTION_FOR_CHATGPT_PRO.md` | not in the repository (its code was not recoverable) | reimplemented in REPORT 7.15 | 7.15 |
| `questions/QUESTION2_FOR_CHATGPT_PRO.md` | `chatgpt_answer_1.md` | `population_estimator.py`, `population_summary.json`, `pop1k_2e24_2e26.npz`, `results_recovered/` | 7.16 |
| `questions/QUESTION3_FOR_CHATGPT_PRO.md` | `chatgpt_answer_2.md` | `roles_round2/`, rerun in `roles_run_2e28/` and `roles_run_2e30/` | 7.16 |
| `questions/QUESTION4_FOR_CHATGPT_PRO.md` | `chatgpt_answer_3.md` | | 7.16 (`z13`–`z39`) |
| `questions/QUESTION5_FOR_CHATGPT_PRO.md` | `chatgpt_answer_4.md` (summary); full text in `round5/answer.md` | `round5/` | 7.16 (`z40`–`z44`) |
| `questions/QUESTION6_FOR_CHATGPT_PRO.md` | `round6/answer.md` | `round6/` | 7.17 (`z48`–`z53`) |
| `questions/QUESTION7_FOR_CHATGPT_PRO.md` | `round7/answer.md` | `round7/`: `round7_estimator.py`, `ROUND7_REPORT.pdf`, `results/`, `runs/` | 7.18 (`z59`–`z61`) |

Run the code from the repository root. Round 5's scripts import `lines` from `scripts/`, so run them with `PYTHONPATH=scripts`.
