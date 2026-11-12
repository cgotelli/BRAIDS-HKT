"""Explicitly identified, private small-cell control on frozen Test1 inputs."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from validation.hydraulics.merged_fwave115 import explicit
from validation.hydraulics.smallcell117 import redistribution
explicit.Explicit=redistribution.Explicit;explicit.install=redistribution.install;explicit.METHOD=redistribution.METHOD
from tools.profile_terrain_amr116 import main
if __name__=='__main__':main()
