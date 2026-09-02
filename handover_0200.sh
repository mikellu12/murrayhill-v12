#!/usr/bin/env bash
# Idle until 02:00, then Murray Hill, then London.
#
# 1  RATE THE 180 STRIPS. Murray Hill's ratings come from svi_90 and carry
#    "Manhattan" in the prompt; London is placeless and half its nodes are
#    180-degree strips. Rating Murray Hill's 180s placeless closes both gaps at
#    once and gives the pedestrian arm a Murray Hill counterpart it has never
#    had -- London's 401 pedestrian nodes currently compare against nothing.
#
#    --mast-set svi_90_wide, NOT the folder name. svi_180 was re-rendered at
#    2880x1833, so the svi_180 calibration -- tuned for the old 1440x916 -- now
#    masks the wrong height and band. The folder-name default would pick it.
#
# 2  ASK IT IN WORDS, on a sample stratified by M. Illustration, not
#    validation: a model asked to justify a rating writes a fluent
#    justification either way.
#
# 3  RESUME THE LONDON RE-ASK, only if the re-rate completed.
set -u
cd "$(dirname "$0")"
mkdir -p logs
GPU=".venv-gpu/Scripts/python.exe"

run() { powershell.exe -NoProfile -Command \
  "Start-Process -FilePath (Resolve-Path '$GPU') -ArgumentList $1 -NoNewWindow -Wait \
   -RedirectStandardOutput '$2' -RedirectStandardError '$2.err'"; }

while :; do [ "$((10#$(date +%H)))" -eq 2 ] && break; sleep 240; done
echo "[$(date '+%F %T')] Murray Hill: rating the 180-degree strips, placeless"

run "'tools/sim_vlm_run.py','--src','data/raw/svi_180','--table','results/tables/sim_vlm_180_placeless.csv','--anchors','7','--mast-set','svi_90_wide'" logs/mh_180_rate.log
R=$(( $(wc -l < results/tables/sim_vlm_180_placeless.csv 2>/dev/null || echo 1) - 1 ))
echo "[$(date '+%F %T')] 180 ratings: $R of 1514"

echo "[$(date '+%F %T')] Murray Hill: qualitative pass"
run "'tools/sim_vlm_describe.py','--src','data/raw/svi_180','--mast-set','svi_90_wide','--n','90','--table','results/tables/vlm_descriptions_180.csv'" logs/mh_180_describe.log

[ "$R" -ge 1514 ] || { echo "180 ratings incomplete -- stopping here"; exit 1; }

echo "[$(date '+%F %T')] Murray Hill: re-rating the 90-degree halves, placeless"
run "'tools/sim_vlm_run.py','--src','data/raw/svi_90','--table','results/tables/sim_vlm_v4_placeless.csv','--anchors','7','--mast-set','svi_90'" logs/mh_rerate.log
R90=$(( $(wc -l < results/tables/sim_vlm_v4_placeless.csv 2>/dev/null || echo 1) - 1 ))
echo "[$(date '+%F %T')] 90 ratings: $R90 of 3064"
[ "$R90" -ge 3064 ] || { echo "90 re-rate incomplete -- NOT starting the re-ask"; exit 1; }

echo "[$(date '+%F %T')] resuming the London re-ask"
powershell.exe -NoProfile -Command \
  "\$e=[System.Environment]; \$e::SetEnvironmentVariable('SIM_CONFIG','config_london.yaml');
   Start-Process -FilePath (Resolve-Path '$GPU') -ArgumentList \
   'tools/sim_vlm_converge.py','--src','data/london/raw/svi_90',\
   '--table','results/london/tables/sim_vlm_london_converged.csv',\
   '--mast-set','svi_90' -NoNewWindow -Wait \
   -RedirectStandardOutput 'logs/london_converge.log' -RedirectStandardError 'logs/london_converge.err'"
echo "[$(date '+%F %T')] all done"
