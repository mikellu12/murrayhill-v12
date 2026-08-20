"""Run the Murray Hill streetscape pipeline.

    python main.py                 all stages
    python main.py --from s04      resume from a stage
    python main.py --only s06 s07  specific stages

Every stage checkpoints and skips completed work, so re-running is cheap.
Stages 1-3 need a GPU and a GMAPS_KEY; 4-8 run on CPU from saved data.
"""
import subprocess, sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
STAGES = ["s01_frame", "s02_imagery", "s03_profiles", "s04_metrics",
          "s05_geometry", "s06_analysis", "s07_enclosure", "s08_figures"]

args = sys.argv[1:]
run = STAGES
if "--from" in args:
    k = args[args.index("--from") + 1]
    run = STAGES[[s.startswith(k) for s in STAGES].index(True):]
elif "--only" in args:
    keys = args[args.index("--only") + 1:]
    run = [s for s in STAGES if any(s.startswith(k) for k in keys)]

for name in run:
    p = SRC / f"{name}.py"
    if not p.exists():
        print(f"skip {name}")
        continue
    r = subprocess.run([sys.executable, str(p)])
    if r.returncode != 0:
        sys.exit(f"{name} failed ({r.returncode})")
print("\ndone -- results/figures and results/tables")
