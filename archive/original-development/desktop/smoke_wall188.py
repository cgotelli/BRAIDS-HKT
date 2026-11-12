"""Public groin with moving wet/dry AMR and exact restart on the new runtime."""
import sys
from pathlib import Path
import check_smart183 as test
ROOT=Path(__file__).resolve().parents[1]
test.OUT=ROOT/'validation/_wall188/smoke'
sys.argv+=['--runtime',str(ROOT/'validation/_wall188/desktop_candidate')]
if '--tag' not in sys.argv:sys.argv+=['--tag','directional']
if __name__=='__main__':test.main()
