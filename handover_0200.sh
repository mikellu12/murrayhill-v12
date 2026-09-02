#!/usr/bin/env bash
# Idle until 02:00, then Murray Hill, then the London re-ask.
#
# ORDER, and why. Murray Hill's ratings are svi_90 and carry "Manhattan" in the
# prompt while London is placeless, so no cross-city number is clean yet. Its
# 180-degree strips go first because they close two gaps at once: London's 401
# pedestrian nodes are 180 strips that currently compare against nothing.
#
# --mast-set svi_180 everywhere. The 180 tree was re-rendered at 2880x1833 and
# the calibration is now named for that geometry.
#
# THE QUALITATIVE PASS COVERS EVERY FRAME, not a sample: a walk-through
# interface shows a verdict at each step, and a sample leaves the sidebar blank
# most of the way down a street. Measured at 4.9 s per frame for two questions,
# so four questions over 1,514 strips is about four hours.
#
# The questions are scene, greenery, ground and frontage -- the last three
# corresponding to I, D and Y. "What stands out" was dropped after the probe
# answered it with the logo on a parked truck: it invites the model to name
# incidental objects rather than anything about the street.
set -u
cd "$(dirname "$0")"
mkdir -p logs
GPU=".venv-gpu/Scripts/python.exe"

run() { powershell.exe -NoProfile -Command \
  "Start-Process -FilePath (Resolve-Path '$GPU') -ArgumentList $1 -NoNewWindow -Wait \
   -RedirectStandardOutput '$2' -RedirectStandardError '$2.err'"; }
rows() { echo $(( $(wc -l < "$1" 2>/dev/null || echo 1) - 1 )); }

while :; do [ "$((10#$(date +%H)))" -eq 2 ] && break; sleep 240; done

echo "[$(date '+%F %T')] 1/4  rating the 180 strips, placeless"
run "'tools/sim_vlm_run.py','--src','data/raw/svi_180','--table','results/tables/sim_vlm_180_placeless.csv','--anchors','7','--mast-set','svi_180'" logs/mh_180_rate.log
echo "[$(date '+%F %T')]      $(rows results/tables/sim_vlm_180_placeless.csv) of 1514"

echo "[$(date '+%F %T')] 2/4  describing every 180 strip"
run "'tools/sim_vlm_describe.py','--src','data/raw/svi_180','--mast-set','svi_180','--all','--fields','scene','greenery','ground','frontage','--table','results/tables/vlm_descriptions_180.csv'" logs/mh_180_describe.log
echo "[$(date '+%F %T')]      $(rows results/tables/vlm_descriptions_180.csv) described"

echo "[$(date '+%F %T')] 3/4  re-rating the 90 halves, placeless"
run "'tools/sim_vlm_run.py','--src','data/raw/svi_90','--table','results/tables/sim_vlm_v4_placeless.csv','--anchors','7','--mast-set','svi_90'" logs/mh_rerate.log
R90=$(rows results/tables/sim_vlm_v4_placeless.csv)
echo "[$(date '+%F %T')]      $R90 of 3064"
[ "$R90" -ge 3064 ] || { echo "incomplete -- NOT starting the re-ask"; exit 1; }

echo "[$(date '+%F %T')] 4/4  resuming the London re-ask"
powershell.exe -NoProfile -Command \
  "\$e=[System.Environment]; \$e::SetEnvironmentVariable('SIM_CONFIG','config_london.yaml');
   Start-Process -FilePath (Resolve-Path '$GPU') -ArgumentList \
   'tools/sim_vlm_converge.py','--src','data/london/raw/svi_90',\
   '--table','results/london/tables/sim_vlm_london_converged.csv',\
   '--mast-set','svi_90' -NoNewWindow -Wait \
   -RedirectStandardOutput 'logs/london_converge.log' -RedirectStandardError 'logs/london_converge.err'"
echo "[$(date '+%F %T')] all done"
