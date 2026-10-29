# Source disclosure and verification — 2026-10-29

Own research source, four reviewed public experiment configurations and tests are included.
Private Git history, all results/runs, weights/factors, token data, archives, logs and runtime settings
are excluded. Source confidentiality/provenance review found no confirmed secret/private-data blocker.

The original `not-supported` conclusion, one-MLP scope and absence of demonstrated speedup remain.
REPORT/VERIFICATION files are dated historical records; their old test counts and measurements
are not new verification of this source copy, nor replayable benchmarks without the omitted artifacts.

PASS — all 20 selected Python modules parsed as AST during the source review.
NOT RUN — Linux/WSL CPU suite: the existing Ubuntu distro failed to start with
`HCS_E_CONNECTION_TIMEOUT`. No WSL reset, runtime repair, model/GPU run or dataset download was attempted.
Native Windows execution is limited by the source's Unix fcntl locking dependency.

On an already prepared Linux environment with the declared dependencies:

```sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python -m pytest -q -k 'not actual_phase1_history_unchanged and not phase1_and_phase2_artifacts_and_history_preserved'
```

The two excluded history tests require intentionally omitted private run artifacts. Do not count
their omission or optional GPU/real-artifact skips as passed checks. This source preview makes no
deployment, performance or scientific-replication acceptance claim.
