"""Re-exercise the historical growing bed eigenmode; no reference overwrite."""
import argparse
import shutil
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'desktop')]
from wave164 import source
from validation.sediment.river_morphodynamics import shoreline_wave_probe as case
from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine
from braids_core.sediment.bend_methods_v1 import digest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['matrix','hll','rusanov','bed_diagonal','morph_llf','capacity_diagonal'],required=True)
    ap.add_argument('--tag',default='v2')
    args=ap.parse_args()
    out=ROOT/'validation/_wave164/stability'/(args.mode+'_'+args.tag);out.mkdir(parents=True,exist_ok=False)
    original=case.OUT/'coarse/input'
    inp=out/'case/coarse/input'
    shutil.copytree(original,inp,ignore=shutil.ignore_patterns('gpu_sources'))
    case.transport.prepare_sources(inp/'gpu_sources',case.transport.read_records(inp),integrator='ssp33')
    case.OUT=out/'case'
    target=out/'wave.h';target.write_text(source((ROOT/'validation/sediment/river_morphodynamics/matrix_wave.h').read_text(),args.mode))
    def install(p):
        p.wave_engine=WaveEngine(p.engine.backend,p.flow.threads,matrix=True,source=target)
        p.identity['wave164_stability']=dict(mode=args.mode,source_sha256=digest(target))
    # The same actually coupled step and perturbations as the recorded
    # transcritical instability; source files and prior output stay immutable.
    case.run(correction=True,matrix=True,installer=install,destination=out)


if __name__=='__main__':main()
