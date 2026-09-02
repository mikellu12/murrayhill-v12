#!/usr/bin/env bash
# Keep the pipeline alive and prove it is moving.
#
# The 02:00 run failed in two seconds and its log still ended "all done",
# because nothing checked exit codes and nothing checked that rows were being
# written. This checks both, every five minutes, and restarts the pipeline if
# it has died with work outstanding. Every stage resumes from its own table, so
# a restart costs at most one checkpoint.
#
# It also records progress to logs/progress.tsv, so "is it running" has an
# answer that is a number rather than a process count.
set -u
REPO="C:/Users/lumic/Documents/murrayhill"
cd "$REPO"
rows() { echo $(( $(wc -l < "$1" 2>/dev/null || echo 1) - 1 )); }
alive() { powershell.exe -NoProfile -Command \
  "(Get-CimInstance Win32_Process | Where-Object { (\$_.CommandLine -like '*sim_vlm_run*' -or \$_.CommandLine -like '*sim_vlm_describe*' -or \$_.CommandLine -like '*sim_vlm_converge*') -and \$_.CommandLine -notlike '*CimInstance*' }).Count" 2>/dev/null | tr -d '\r\n '; }

STALL=0
while :; do
  A=$(rows results/tables/sim_vlm_180_placeless.csv)
  B=$(rows results/tables/vlm_descriptions_180.csv)
  C=$(rows results/london/tables/sim_vlm_london_converged.csv)
  N=$(alive); N=${N:-0}
  printf "%s\t180=%s/1514\tdesc=%s/1514\treask=%s/6422\tprocs=%s\n" \
    "$(date '+%F %T')" "$A" "$B" "$C" "$N" >> logs/progress.tsv

  DONE=0
  [ "$A" -ge 1514 ] && [ "$B" -ge 1514 ] && [ "$C" -ge 6422 ] && DONE=1
  if [ "$DONE" = "1" ]; then
    echo "$(date '+%F %T') all stages complete" >> logs/progress.tsv
    break
  fi

  if [ "$N" -eq 0 ]; then
    STALL=$((STALL+1))
    # two consecutive idle checks, so a handover between stages is not mistaken
    # for a death
    if [ "$STALL" -ge 2 ]; then
      echo "$(date '+%F %T') RESTARTING: nothing running, work outstanding" >> logs/progress.tsv
      S=$(date +%H%M%S); cp handover_0200.sh "/tmp/hand_$S.sh"
      nohup bash "/tmp/hand_$S.sh" >> logs/handover.log 2>&1 &
      STALL=0
      sleep 120
    fi
  else
    STALL=0
  fi
  sleep 300
done
