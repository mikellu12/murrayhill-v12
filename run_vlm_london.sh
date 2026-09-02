#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
export SIM_CONFIG=config_london.yaml
powershell.exe -NoProfile -Command \
 "\$e=[System.Environment]; \$e::SetEnvironmentVariable('SIM_CONFIG','config_london.yaml');
  Start-Process -FilePath '.venv-gpu/Scripts/python.exe' -ArgumentList \
   'tools/sim_vlm_run.py','--src','data/london/raw/svi_90',\
   '--table','results/london/tables/sim_vlm_london.csv','--anchors','7',\
   '--mast-set','svi_90' -NoNewWindow -Wait \
   -RedirectStandardOutput 'logs/london_vlm.log' -RedirectStandardError 'logs/london_vlm.err'"
echo "[$(date '+%H:%M:%S')] London VLM finished"
