"""Halved hydraulic CFL continuation; same forcing and saved initial fields."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
import check_trace179 as check
from face179 import install


def main(backend,variant):
    original_setup=check.setup
    def setup():
        original_setup()
        if variant=='facewise':install(backend)
    check.setup=setup
    original=check.document
    def document(*args,**kwargs):
        doc=original(*args,**kwargs);doc['settings']['controls']['cfl']=.5;return doc
    check.document=document
    check.main(backend,'run',variant+'_half')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('backend',choices=('cpu','gpu'))
    p.add_argument('variant',choices=('baseline','facewise'));a=p.parse_args();main(a.backend,a.variant)
