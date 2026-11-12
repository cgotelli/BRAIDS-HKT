"""Public Smart-Jaeggi groin regression on the new independent-inflow contract."""
import sys
from pathlib import Path
import check_smart183 as smoke
ROOT=Path(__file__).resolve().parents[1]
smoke.OUT=ROOT/'validation/_inlet187/smoke'
sys.argv+=['--runtime',str(ROOT/'validation/_inlet187/prescribed_candidate')]
if '--tag' not in sys.argv:sys.argv+=['--tag','prescribed']
smoke.main()
