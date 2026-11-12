"""Recompile shared constitutive and wave sources; preserve hydraulic binaries."""
import stage_camenen192 as stage
from stage_bulk213 import ROOT,BASE,OUT,TARGET,BASE_HASH
stage.BASE,stage.OUT,stage.TARGET,stage.BASE_HASH=BASE,OUT,TARGET,BASE_HASH
from build_camenen192 import main
if __name__=='__main__':main()
