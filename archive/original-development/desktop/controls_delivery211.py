"""Eight ordinary laws remain unchanged by the Jiang-Cao release."""
import sys
from pathlib import Path
import check_camenen192 as driver
driver.OUT=Path(__file__).resolve().parents[1]/'validation/_delivery211'
sys.argv += ['--exclude-law','mpm_jc2026_p15_reconstruction_v1']
driver.main()
