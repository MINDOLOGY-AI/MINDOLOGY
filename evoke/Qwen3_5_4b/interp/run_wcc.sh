#!/bin/bash
# the unattended weakly causal crosscoder run on an 8-gpu box (dataset already tokenized, Qwen weights downloaded).
# run from repo root in tmux: tmux new -d -s wcc "evoke/Qwen3_5_4b/interp/run_wcc.sh <features per layer> <final lambda>"
# logs/wcc.log (appended), logs/status = DONE or FAILED:wcc. a rerun resumes from the latest checkpoint.
set -uo pipefail
mkdir -p logs
rm -f logs/status
echo wcc > logs/phase
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True venv/bin/torchrun --nproc_per_node 8 -m evoke.Qwen3_5_4b.interp.wcc_resid_jumprelu "$1" "$2" 2>&1 | tee -a logs/wcc.log
if [ "${PIPESTATUS[0]}" -eq 0 ]; then echo DONE > logs/status; else echo FAILED:wcc > logs/status; fi
