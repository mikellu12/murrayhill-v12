#!/usr/bin/env bash
# London, clean: one placeless rating pass, then the elimination re-ask.
set -u
cd "$(dirname "$0")"
mkdir -p logs results/london/tables
export SIM_CONFIG=config_london.yaml
step() { echo; echo "=== $* ==="; date '+%m-%d %H:%M:%S'; }

gpu() {
  local log="$1"; shift
  powershell.exe -NoProfile -Command \
    "Start-Process -FilePath '.venv-gpu/Scripts/python.exe' -ArgumentList '$*' \
       -NoNewWindow -Wait -RedirectStandardOutput '$log' \
       -RedirectStandardError '$log.err'"
}

step "rating pass, single, placeless prompt"
gpu "logs/london_vlm.log" tools/sim_vlm_run.py --src data/london/raw/svi_90 \
    --table results/london/tables/sim_vlm_london.csv --anchors 7 \
    --mast-set svi_90
R=$(( $(wc -l < results/london/tables/sim_vlm_london.csv) - 1 ))
echo "rated $R of 6422"
[ "$R" -ge 6422 ] || { echo "ABORT: rating pass incomplete, not starting the re-ask"; exit 1; }

step "M from the single pass"
.venv/Scripts/python tools/sim_compute.py \
    --table results/london/tables/sim_vlm_london.csv \
    > logs/london_sim.log 2>&1
grep -E "^  M |^  M_local|A_i|street type" logs/london_sim.log | head

# The re-ask. Every field pruned to the rungs above chance and asked again
# among the survivors, until one rung is left. Queued behind the single pass
# rather than beside it: both want the whole GPU, and the single pass is what
# the presentation needs.
step "re-ask by elimination"
gpu "logs/london_converge.log" tools/sim_vlm_converge.py \
    --src data/london/raw/svi_90 \
    --table results/london/tables/sim_vlm_london_converged.csv \
    --mast-set svi_90
echo "re-ask rows: $(( $(wc -l < results/london/tables/sim_vlm_london_converged.csv 2>/dev/null || echo 1) - 1 ))"
step "queue complete"
