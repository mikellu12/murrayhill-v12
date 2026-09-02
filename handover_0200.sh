#!/usr/bin/env bash
# 02:00. Murray Hill on the 180 strips, then the London re-ask. The 90-degree
# re-rate runs last and only if everything before it finished.
#
# Same ten fields, same seven rungs, same placeless prompt. --mast-set svi_180
# throughout: that tree is 2880x1833 now and the calibration is named for it.
#
# THE QUALITATIVE PASS COVERS EVERY FRAME. A walk-through interface shows a
# verdict at each step, so a sample leaves the sidebar blank most of the way
# down a street. Four questions -- scene, greenery, ground, frontage -- the last
# three matching imageability, dependence and identity so the text lines up
# with the scores beside it.
#
# The question wording was tested and is not arbitrary. Generation is
# deterministic here (five frames, byte-identical across runs), but the ANSWER
# IS SENSITIVE TO PHRASING: constraining the system prompt to two sentences
# flipped a factual claim about greenery on a facade. So the wording is fixed
# and recorded, and the text stays an illustration -- the measured twins remain
# the evidence.
#
# Earlier wordings failed in ways worth not repeating: "what stands out" was
# answered with the logo on a parked truck, an unconstrained "describe the
# ground plane" ran to a numbered list and truncated mid-word, and "describe
# the frontage" described the truck rather than the buildings.
set -u
# ABSOLUTE, not $(dirname "$0"). This script is copied to /tmp before launch --
# bash reads a script by byte offset, so editing one in place while it runs
# makes it execute garbage from mid-line -- and from /tmp, dirname is /tmp.
# That is what killed the 02:00 run: Resolve-Path '.venv-gpu/...' returned
# empty, Start-Process rejected a null FilePath, and every step failed in two
# seconds while the log still ended with "all done".
cd "C:/Users/lumic/Documents/murrayhill"
mkdir -p logs
GPU=".venv-gpu/Scripts/python.exe"
run() { powershell.exe -NoProfile -Command \
  "Start-Process -FilePath (Resolve-Path '$GPU') -ArgumentList $1 -NoNewWindow -Wait \
   -RedirectStandardOutput '$2' -RedirectStandardError '$2.err'"; }
rows() { echo $(( $(wc -l < "$1" 2>/dev/null || echo 1) - 1 )); }

if [ "${WAIT_FOR_0200:-0}" = "1" ]; then
  while :; do [ "$((10#$(date +%H)))" -eq 2 ] && break; sleep 240; done
fi

echo "[$(date '+%F %T')] 1/4  scores, 180 strips, placeless"
run "'tools/sim_vlm_run.py','--src','data/raw/svi_180','--table','results/tables/sim_vlm_180_placeless.csv','--anchors','7','--mast-set','svi_180'" logs/mh_180_rate.log
R=$(rows results/tables/sim_vlm_180_placeless.csv)
echo "[$(date '+%F %T')]      $R of 1514"

echo "[$(date '+%F %T')] 2/4  qualitative, every 180 strip, four questions"
run "'tools/sim_vlm_describe.py','--src','data/raw/svi_180','--mast-set','svi_180','--all','--fields','scene','greenery','ground','frontage','--table','results/tables/vlm_descriptions_180.csv'" logs/mh_180_describe.log
echo "[$(date '+%F %T')]      $(rows results/tables/vlm_descriptions_180.csv) described"

echo "[$(date '+%F %T')] 3/4  London re-ask"
powershell.exe -NoProfile -Command \
  "\$e=[System.Environment]; \$e::SetEnvironmentVariable('SIM_CONFIG','config_london.yaml');
   Start-Process -FilePath (Resolve-Path '$GPU') -ArgumentList \
   'tools/sim_vlm_converge.py','--src','data/london/raw/svi_90',\
   '--table','results/london/tables/sim_vlm_london_converged.csv',\
   '--mast-set','svi_90' -NoNewWindow -Wait \
   -RedirectStandardOutput 'logs/london_converge.log' -RedirectStandardError 'logs/london_converge.err'"
echo "[$(date '+%F %T')]      re-ask at $(rows results/london/tables/sim_vlm_london_converged.csv) of 6422"

# Only if the machine is otherwise done. The 90 re-rate is what would put the
# two cities on one instrument, but the study is moving to the 180 render, so
# it waits behind everything that the interface and the re-ask need.
echo "[$(date '+%F %T')] 4/4  idle work: re-rating the 90 halves"
run "'tools/sim_vlm_run.py','--src','data/raw/svi_90','--table','results/tables/sim_vlm_v4_placeless.csv','--anchors','7','--mast-set','svi_90'" logs/mh_rerate.log
echo "[$(date '+%F %T')] all done"
