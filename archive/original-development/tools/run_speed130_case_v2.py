"""Run the existing natural-case harness with the explicit dry admission audit."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import research_wet_cfl130, research_wet_cfl130_v2, run_speed130_case

if __name__=='__main__':
    output=Path(sys.argv[sys.argv.index('--output')+1])
    before=research_wet_cfl130.candidate
    research_wet_cfl130.candidate=research_wet_cfl130_v2.candidate
    try:run_speed130_case.main()
    finally:
        research_wet_cfl130.candidate=before
        if output.exists():
            (output/'DRY_ADMISSION_OVERRIDE.json').write_text(json.dumps(dict(
                research_only=True, scripts={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                for p in ('tools/run_speed130_case_v2.py','tools/research_wet_cfl130_v2.py')},
                replaced_check='Only dry==1e-6 qualification admission; actual native and physical thresholds asserted equal'),indent=2))
