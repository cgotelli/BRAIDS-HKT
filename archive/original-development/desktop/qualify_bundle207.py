"""Bounded Desktop integration campaign; no published-flume claim."""
import argparse,json,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--backend',choices=['cpu','gpu'],default='cpu')
    p.add_argument('--scenario',choices=['front','bank','open','oblique'],default='front');p.add_argument('--capture',action='store_true')
    p.add_argument('--runtime');p.add_argument('--wet-start',action='store_true');p.add_argument('--finest',action='store_true')
    args=p.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    if args.runtime:sys.path.insert(0,str(Path(args.runtime).resolve()))
    if args.capture:
        from braids_core.sediment.portable_river_runtime import install
        install()
    elif args.runtime:
        from braids_core.river_launcher import verify
        from braids_core.sediment.portable_river_runtime import install
        verify();install(json.loads((Path(args.runtime)/'PRECOMPILED.json').read_text()),Path(args.runtime).resolve())
        import subprocess
        original=subprocess.Popen
        def no_compiler(command,*a,**kw):
            exe=str(command[0] if isinstance(command,(list,tuple)) else command).lower()
            if any(v in exe for v in ('gfortran','g++','gcc','clang')):raise AssertionError('Runtime compilation: '+exe)
            return original(command,*a,**kw)
        subprocess.Popen=no_compiler
    from test_jiang_cao207 import project,request
    from braids_core.river_model import resolve_model
    from braids_core.river_project import validate_project
    from braids_desktop.preparation import prepare
    from braids_core.river_worker import run
    from braids_core.river_open_boundary import resolve
    from braids_core.boundaries.domain import PolygonDomain
    from braids_desktop.services import geometry_grid
    import numpy as np
    from osgeo import gdal
    doc=project(out/'input',args.backend,adaptive=args.scenario!='open',collapse=args.scenario=='bank',
                bank=args.scenario=='bank',opened=args.scenario!='bank')
    if args.scenario in ('front','oblique'):
        doc['initial'].update(stage_m=0.,velocity_m_s=[0.,0.])
        # Same open ends; lengthen the reach, retaining aligned inlet/outlet.
        dom=doc['settings']['domain'];dom['vertices_xy']=[[0,0],[3.2,0],[3.2,.8],[0,.8]]
        if args.scenario=='oblique':dom['vertices_xy']=[[0,0],[3.2,.2],[3.2,1.],[0,.8]]
        domain=PolygonDomain(**dom)
        doc['settings']['grid']=geometry_grid(domain,.2,2)
        doc['settings']['assignments']=[]
        for x,name,kind in ((0,'inlet','inlet'),(3.2,'outlet','outlet')):
            e=next(e for e in domain.edges if e.start_xy[0]==e.end_xy[0]==x)
            doc['settings']['assignments'].append(dict(edge_id=e.edge_id,bc_id=name,kind=kind))
        doc['settings']['hydraulics']['outlet']=dict(method='outlet.stage',stage_m=0.)
        doc['settings'].update(end_time_s=.8,output_interval_s=.1,checkpoint_interval_s=.4,dt_max_s=.002)
        doc['settings']['adaptation'].update(interval=.05,buffer_cells=1)
        if args.wet_start:doc['initial'].update(stage_m=.15,velocity_m_s=[.2,0.])
        if args.finest:
            doc['settings']['levels']=1;doc['settings']['grid']=geometry_grid(domain,.2,1)
            doc['settings']['adaptive']=False;doc['settings']['adaptation']['terrain_enabled']=True
            doc['settings']['end_time_s']=.2;doc['settings']['checkpoint_interval_s']=.1
    elif args.scenario=='bank':
        m=request(args.backend,True);m['sediment']['selection']['capacity']['parameters']['ProtectInitialDryBank']=1
        doc['model']=resolve_model(m).document()
        doc['settings'].update(end_time_s=.16,output_interval_s=.02,checkpoint_interval_s=.08)
    else:
        doc['settings'].update(end_time_s=.08,output_interval_s=.02,checkpoint_interval_s=.04)
    kinds={a['bc_id']:a['kind'] for a in doc['settings']['assignments']}
    doc['boundary_resolution']=resolve(doc['settings']['hydraulics'],doc['boundary_states'],kinds,
        sediment=doc['model']['physics']['sediment'],ng=1,end_time=doc['settings']['end_time_s'])
    doc=validate_project(doc);project_file=out/'project.json';project_file.write_text(json.dumps(doc,indent=2),encoding='utf-8')
    prep=prepare(project_file,out/'prepared',asset_cache=out/'assets')
    result=run(prep,out/'run',checkpoint_dir=out/'checkpoints')
    report=dict(scenario=args.scenario,backend=args.backend,result=result,compiler_free=bool(args.runtime))
    if args.runtime:
        import hashlib
        report['runtime']=hashlib.sha256((Path(args.runtime)/'RUNTIME.json').read_bytes()).hexdigest()
        for name in ('braids_core.river_worker','braids_core.sediment.jiang_cao_desktop','braids_desktop.preparation'):
            assert Path(sys.modules[name].__file__).resolve().is_relative_to(Path(args.runtime).resolve()),name
    if args.capture:
        from braids_core.sediment.portable_river_runtime import RECORDS
        (out/'NATIVE_RECORDS.json').write_text(json.dumps(RECORDS,indent=2),encoding='utf-8')
    (out/'CHECK.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],time=result['time_s'],steps=result['steps'],
                         changed=sum(e['changed'] for e in result['adaptation_events']))),flush=True)


if __name__=='__main__':main()
