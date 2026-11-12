"""Retest the supported Smart-Jaeggi route with advanced physics off."""
import sys
from stage_secondary185 import OUT,TARGET
import check_smart183 as smoke
smoke.OUT=OUT/'smoke'
sys.argv += ['--runtime',str(TARGET)]
if '--tag' not in sys.argv:sys.argv+=['--tag','release_v6']
smoke.main()
