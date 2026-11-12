"""Compose the isolated 2D CFL tests with the identified predictor experiment."""
from contextlib import contextmanager
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import research_wet_cfl130 as numerical
from tools import research_2d_cfl130 as run
from validation.hydraulics.speed130_clock import candidate as clock_candidate


if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--clock',default='combined')
    a,remaining=p.parse_known_args();sys.argv=[sys.argv[0],*remaining]
    original=numerical.candidate
    @contextmanager
    def candidate(cap,dry=1e-6):
        with original(cap,dry) as context:
            with clock_candidate(a.clock) as receipt:
                context['identity']['clock']=receipt
                yield context
    numerical.candidate=candidate
    try:run.main()
    finally:numerical.candidate=original
