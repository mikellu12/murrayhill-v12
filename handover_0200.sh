#!/usr/bin/env bash
# Everything is stopped until 02:00. Then Murray Hill re-rates, and only when
# it finishes does the London re-ask resume.
#
# ORDER IS THE POINT. Until Murray Hill is rated with the same placeless prompt
# as London the two cities are not on one instrument, so every cross-city
# number is provisional -- that is the claim a reviewer presses on. The re-ask
# is a robustness check on a result that already exists, and it checkpoints
# every 25 images, so it waits at no cost but time.
#
# The re-rate writes a NEW table. sim_vlm_v3.csv is the "Manhattan" prompt run
# and every figure and validation number already presented comes from it;
# overwriting it would silently change shown work.
set -u
cd "$(dirname "$0")"
mkdir -p logs

# wait for 02:00 -- compared as a number so 0159 < 0200 <= 0259 is unambiguous
while :; do
  h=$(date +%H); m=$(date +%M)
  [ "$((10#$h))" -eq 2 ] && break
  sleep 240
done
echo "[$(date '+%F %T')] starting Murray Hill re-rate"

powershell.exe -NoProfile -Command \
  "Start-Process -FilePath (Resolve-Path .venv-gpu/Scripts/python.exe) -ArgumentList \
   'tools/sim_vlm_run.py','--src','data/raw/svi_90',\
   '--table','results/tables/sim_vlm_v4_placeless.csv','--anchors','7',\
   '--mast-set','svi_90' -NoNewWindow -Wait \
   -RedirectStandardOutput 'logs/mh_rerate.log' -RedirectStandardError 'logs/mh_rerate.err'"

R=$(( $(wc -l < results/tables/sim_vlm_v4_placeless.csv 2>/dev/null || echo 1) - 1 ))
echo "[$(date '+%F %T')] Murray Hill re-rate finished: $R of 3064 rows"
[ "$R" -ge 3064 ] || { echo "incomplete -- NOT starting the re-ask"; exit 1; }

echo "[$(date '+%F %T')] resuming the London re-ask"
export SIM_CONFIG=config_london.yaml
powershell.exe -NoProfile -Command \
  "\$e=[System.Environment]; \$e::SetEnvironmentVariable('SIM_CONFIG','config_london.yaml');
   Start-Process -FilePath (Resolve-Path .venv-gpu/Scripts/python.exe) -ArgumentList \
   'tools/sim_vlm_converge.py','--src','data/london/raw/svi_90',\
   '--table','results/london/tables/sim_vlm_london_converged.csv',\
   '--mast-set','svi_90' -NoNewWindow -Wait \
   -RedirectStandardOutput 'logs/london_converge.log' -RedirectStandardError 'logs/london_converge.err'"
echo "[$(date '+%F %T')] London re-ask finished"
