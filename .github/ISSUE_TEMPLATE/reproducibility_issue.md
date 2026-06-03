---
name: Reproducibility issue
about: A paper figure, table, or §V number doesn't reproduce.
title: "[REPRO] "
labels: ["reproducibility"]
assignees: []
---

## Which experiment

<!-- Pick one (or list multiple). The §V research-question label is
what the paper uses; the runner module is the code path. -->

- RQ: <!-- e.g., RQ1 headline / RQ3 close analogues / Ablation budget -->
- Runner: <!-- e.g., experiments.runners.rq1_headline -->
- Config: <!-- e.g., experiments/configs/rq1_headline.yaml -->
- Figure / table / CSV affected: <!-- e.g., results/figures/rq1_headline.pdf -->

## Which commit / version

- Commit hash (from `git rev-parse HEAD`):
- `slack-certify-mapf` version (from `pip show slackcertify`):
- Paper version (camera-ready / arXiv vN / preprint):

## Expected vs observed

<!-- Quote the paper number, then your reproduction's number. Include
a confidence interval if available. -->

- Paper reports: <!-- e.g., success rate 0.92 ± 0.02 at n=100 -->
- I observe: <!-- e.g., 0.78 ± 0.03 -->

## Exact command

<!-- The full command(s) used. Include any non-default flags. -->

```bash
# e.g.:
bash scripts/repro_paper.sh
# or, for a single RQ:
python -m experiments --config experiments/configs/rq1_headline.yaml \
    --output results/raw/rq1_headline.csv \
    --manifest results/manifests/rq1_headline.jsonl
python analysis/make_figures.py
```

## Environment

- OS: <!-- Ubuntu 22.04 / macOS 14 / ... -->
- Python version:
- Solver binaries (paste `python scripts/check_environment.py`):

```
<paste here>
```

- Benchmark suite version (commit of `benchmarks/`, if applicable):
- Hardware (CPU model, RAM, GPU if used):

## What I've checked

- [ ] Same scen_seeds as the paper config
- [ ] Same Δ / ε / p_d as the paper config
- [ ] Solver binaries match commit hashes pinned in `third_party/README.md`
- [ ] No CSV manifest leftover from a prior partial run

## Additional context
