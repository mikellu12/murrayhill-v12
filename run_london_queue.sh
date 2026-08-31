#!/usr/bin/env bash
# London end-to-end, queued behind the s02 fetch, then the Murray Hill
# svi_180 re-render. Each stage logs to logs/ and refuses to start if the
# stage it depends on left nothing behind.
set -u
cd "$(dirname "$0")"
mkdir -p logs
PY=.venv/Scripts/python
GPU=.venv-gpu/Scripts/python.exe
export SIM_CONFIG=config_london.yaml
step() { echo; echo "=== $* ==="; date '+%H:%M:%S'; }

# GPU work is launched detached through PowerShell. A GPU child of this shell
# gets reaped while the model is still loading -- twenty minutes of weights
# read from disk with nothing to show for it. Start-Process puts it outside
# this process tree; -Wait keeps the queue ordered anyway.
gpu() {
  local log="$1"; shift
  powershell.exe -NoProfile -Command \
    "\$e=[System.Environment]; \$e::SetEnvironmentVariable('SIM_CONFIG','config_london.yaml');
     Start-Process -FilePath '$GPU' -ArgumentList '$*' -NoNewWindow -Wait \
       -RedirectStandardOutput '$log' -RedirectStandardError '$log.err'"
}

step "wait for the imagery fetch"
while [ ! -f data/london/processed/manifest.csv ]; do sleep 30; done
# manifest.csv is written last, but give the writer a moment to flush
sleep 5
echo "manifest: $(wc -l < data/london/processed/manifest.csv) rows"

step "render, split by street type"
$PY tools/export_svi_90.py --out data/london/raw/svi_90 \
    > logs/london_render.log 2>&1
tail -4 logs/london_render.log
H=$(find data/london/raw/svi_90 -name '*_[LR].jpg' | wc -l)
F=$(find data/london/raw/svi_90 -name '*_F.jpg' | wc -l)
echo "rendered: $H halves, $F wide strips"
[ "$H" -gt 0 ] || { echo "ABORT: nothing rendered"; exit 1; }

step "validate the mast against London imagery"
$PY tools/mast_calibrate.py --src data/london/raw/svi_90 \
    --pattern '*_L.jpg' --name svi_90 --n 300 2>&1 | tail -12
if [ "$F" -gt 0 ]; then
  $PY tools/mast_calibrate.py --src data/london/raw/svi_90 \
      --pattern '*_F.jpg' --name svi_90_wide --n 300 2>&1 | tail -12
fi

step "segment"
gpu "logs/london_seg.log" tools/seg_two_model.py --src data/london/raw/svi_90 \
    --out data/london/processed/seg90_two_model.csv --mast-set svi_90
tail -5 logs/london_seg.log 2>/dev/null

step "rate -- single pass, full p1..p7 kept for the readout"
mkdir -p results/london/tables
gpu "logs/london_vlm.log" tools/sim_vlm_run.py --src data/london/raw/svi_90 \
    --table results/london/tables/sim_vlm_london.csv --anchors 7 \
    --mast-set svi_90 &
VLM=$!

# CPU-only, so it runs alongside the VLM rather than delaying it. The old tree
# is 1440x916 from a superseded default; clearing it first, because nothing
# renames on a size change and two geometries in one folder is the trap.
step "re-render Murray Hill svi_180 at 2880 (concurrent, CPU)"
( unset SIM_CONFIG
  rm -rf data/raw/svi_180
  .venv/Scripts/python tools/export_svi_180.py --out data/raw/svi_180 \
      > logs/mh_180_rerender.log 2>&1
  echo "svi_180 re-render done: $(find data/raw/svi_180 -name '*.jpg' | wc -l) images"
) &
MH=$!

wait $MH; echo "[$(date '+%H:%M:%S')] Murray Hill svi_180 re-render finished"
wait $VLM; echo "[$(date '+%H:%M:%S')] London VLM finished"
step "queue complete"
