"""Use the same cold/public-worker matrix as inlet187 on the new limiter."""
import json,sys
from pathlib import Path
import qualify_inlet187 as test
ROOT=Path(__file__).resolve().parents[1]
test.OUT=ROOT/'validation/_wall188/release'
test.TARGET=ROOT/'validation/_wall188/desktop_candidate'
sys.path.insert(0,str(test.TARGET))
# The reused driver checks an immutable stage receipt in its output directory.
from stage_secondary185 import save
receipt=json.loads((test.TARGET.parent/'DELIVERY_STAGE.json').read_text())
dest=test.OUT/'PRESCRIBED_STAGE.json'
if dest.exists():assert json.loads(dest.read_text())==receipt
else:save(dest,receipt)
if __name__=='__main__':test.main()
