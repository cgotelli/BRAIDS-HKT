"""Reuse the frozen209 cropped-front input generator with candidate210 only."""
import runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUNTIME=ROOT/'validation/_exchange210/candidate'
module=runpy.run_path(str(ROOT/'desktop/front_shallow209.py'))
module['main'].__globals__['RUNTIME']=RUNTIME
sys.path.insert(0,str(RUNTIME))
module['main']()
