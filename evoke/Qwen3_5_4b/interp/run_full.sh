#!/bin/bash
# the full unattended Qwen3.5-4B SAE run on a gpu box: download the interp dataset -> tokenize -> sae_resid_topk gpu phase.
# run from repo root in tmux: tmux new -d -s run evoke/Qwen3_5_4b/interp/run_full.sh
# logs/<step>.log per step (appended), logs/phase = the current step, logs/status = DONE or FAILED:<step>.
# every step resumes from what is on disk, so a rerun continues where a failed one stopped.
set -uo pipefail
mkdir -p logs
rm -f logs/status
trap 'echo "FAILED:$(cat logs/phase)" > logs/status' ERR
set -e
echo download > logs/phase
venv/bin/python -u -m datasteps.interp_dataset.download_all 2>&1 | tee -a logs/download.log
echo tokenize > logs/phase
venv/bin/python -u -m datasteps.interp_dataset.tokenize_interp_dataset 2>&1 | tee -a logs/tokenize.log
echo gpu > logs/phase
venv/bin/python -u -m evoke.Qwen3_5_4b.interp.sae_resid_topk gpu 2>&1 | tee -a logs/gpu.log
echo DONE > logs/status
