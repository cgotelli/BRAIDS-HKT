"""Cold, compiler-free matched dry groin for timestep qualification."""
import sys
from pathlib import Path
import check_smart183 as driver
ROOT=Path(__file__).resolve().parents[1]
driver.OUT=ROOT/'validation/_step190/runs'
if __name__=='__main__':driver.main()
