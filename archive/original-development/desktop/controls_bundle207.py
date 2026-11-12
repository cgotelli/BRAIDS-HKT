"""Unchanged eight-formula trajectories against retained installed preview34."""
import sys
from pathlib import Path
import check_camenen192 as driver
driver.OUT=Path(__file__).resolve().parents[1]/'validation/_bundle207'
sys.argv += ['--exclude-law','mpm_jc2026_p15_reconstruction_v1']
driver.main()
