"""Public dry-start/regridding regression for each admitted bedload law."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import check_smart183 as driver
driver.OUT=ROOT/'validation/_bend189/dry'
sys.argv+=['--runtime',str(ROOT/'validation/_wall188/desktop_candidate'),'--tag','groin','--dry-amr']
if __name__=='__main__':driver.main()
