"""Execute the installed preview30 payload, not the staged candidate."""
import sys
from pathlib import Path
import check_smart183 as check
ROOT=Path(__file__).resolve().parents[1]
check.OUT=ROOT/'validation/_wall188/installed'
sys.argv+=['--runtime',str(ROOT/'desktop/runtime'),'--tag','public']
if __name__=='__main__':check.main()
