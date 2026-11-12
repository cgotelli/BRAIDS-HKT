"""Aligned full-cell controls that actually activate the isolated wall scheme.

The earlier controls are retained cropped-wall fallback regressions. Here the
angle labels refer to incident velocity/bed orientation, not wall orientation.
"""
import argparse,json
from pathlib import Path
from check_incident181 import setup_factory
from stage_wall172 import save
import control_trace179 as check
ROOT=Path(__file__).resolve().parents[1]

def main(backend):
    original=check.setup;check.setup=lambda:setup_factory(original,backend)
    check.install=lambda backend:None
    check.BASE=ROOT/'validation/_incident181/active_controls'
    document0=check.document
    def document(folder,backend,angle,case):
        from braids_desktop import services
        result=document0(folder,backend,angle,case)
        path=folder/'project/braids_project.json';d=json.loads(path.read_text())
        fresh=services.default_document(folder/'terrain.tif',[[2.,4.],[12.,4.],[12.,6.],[2.,6.]],
            'EPSG:2056',backend=backend,minimum_size_m=.5,levels=1,end_time_s=4.,name='Active wall reconstruction control')
        for key in ('domain','grid'):d['settings'][key]=fresh['settings'][key]
        save(path,d);return result
    check.document=document;check.main(backend)
    folder=check.BASE/'controls'/backend
    for path in folder.glob('*/OWNER.json'):
        assert json.loads(path.read_text())['wallstate181']['eligible_cells']>0,path
    save(folder/'SCOPE181.json',dict(walls='axis-aligned, full square cells',
        angle_labels='incident velocity and bed direction only',candidate_active=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('backend',choices=('cpu','gpu'));main(p.parse_args().backend)
