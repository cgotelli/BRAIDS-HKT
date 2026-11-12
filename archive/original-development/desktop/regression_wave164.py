"""Short historical stability/bend controls with newly authenticated sources."""
import argparse
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'desktop')]
from wave164 import source
from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine
from braids_core.sediment.bend_methods_v1 import digest


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('case',choices=['bend','shoal'])
    ap.add_argument('--backend',choices=['cpu','gpu'],required=True)
    ap.add_argument('--mode',default='capacity_diagonal',choices=['matrix','capacity_diagonal'])
    ap.add_argument('--half-step',action='store_true')
    ap.add_argument('--quarter-step',action='store_true')
    args=ap.parse_args()
    if args.half_step and args.quarter_step:ap.error('Choose one timestep refinement')
    suffix='_quarter' if args.quarter_step else ('_half' if args.half_step else '')
    out=ROOT/f'validation/_wave164/regression/{args.case}_{args.mode}_{args.backend}{suffix}'
    out.mkdir(parents=True,exist_ok=False)
    target=out/'wave.h'
    target.write_text(source((ROOT/'validation/sediment/river_morphodynamics/matrix_wave.h').read_text(),args.mode))
    def install(p):
        p.wave_engine=WaveEngine(p.engine.backend,p.flow.threads,matrix=True,source=target)
        p.identity['wave164_regression']=dict(mode=args.mode,source_sha256=digest(target))
    if args.case=='bend':
        from validation.sediment.river_morphodynamics import wave_bend_regression as case
        case.OUT=out/'fresh_input'
        case.run(args.backend,matrix=True,installer=install,destination=out/'run')
    else:
        from validation.sediment.river_morphodynamics import shoreline_diagnostic as case
        original=case.OUT/'coarse'
        inp=out/'case/coarse/input'
        shutil.copytree(original/'input',inp,ignore=shutil.ignore_patterns('gpu_sources'))
        case.transport.prepare_sources(inp/'gpu_sources',case.transport.read_records(inp),integrator='ssp33')
        warm=original/'cpu/frames.npz'
        case.OUT=out/'case'
        case.run('coarse',args.backend,mobile=True,warm=True,correction=True,matrix=True,
                 installer=install,destination=out/'run',end_time=28.,warm_source=warm,
                 dt_max=.000625 if args.quarter_step else (.00125 if args.half_step else .0025))


if __name__=='__main__':main()
