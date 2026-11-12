"""Installed-runtime groin hydraulic boundary qualification (no solver patches)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.environ.get('BRAIDS_AUDIT_RUNTIME',str(ROOT / 'desktop/runtime'))).resolve()
sys.path[:0] = [str(RUNTIME), str(ROOT)]
sys.dont_write_bytecode = True
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from osgeo import osr
osr.UseExceptions()
import numpy as np
from PySide6.QtWidgets import QApplication
from braids_core.river_launcher import verify
from braids_core.sediment.portable_river_runtime import install
from braids_core.river_open_boundary import HYDRAULICS, new_state, resolve
from braids_core.sediment.bed_columns_v101 import host
from braids_desktop import services
from braids_desktop.demo import create_demo
from braids_desktop.forms import SettingsPanel
from braids_desktop.river_physics import set_choice
from braids_core.boundaries.domain import PolygonDomain

CASES = ('total', 'stage', 'depth', 'critical', 'normal',
         'dry_stage', 'dry_depth', 'dry_critical', 'dry_normal',
         'out_critical', 'out_weir', 'out_supercritical', 'backflow',
         'closed_stage', 'closed_weir', 'schedule', 'zero',
         'invalid_dry_total', 'invalid_supercritical',
         'bedload_normal_critical', 'bedload_stage_weir', 'weir_recovered', 'supercritical_valid',
         'free_supercritical', 'weir_valid', 'bedload_weir_valid')


def document(folder, backend, case):
    end = 120.
    if case in ('out_supercritical', 'backflow', 'closed_stage', 'closed_weir', 'zero'):
        end = 5.
    if case in ('supercritical_valid','free_supercritical'):end=2.
    if case in ('weir_valid','bedload_weir_valid'):end=10.
    if case.startswith('invalid_'): end = .1
    if case == 'schedule': end = 40.
    sediment = case.startswith('bedload_')
    create_demo(folder, backend, end_time_s=end)
    doc = services.load_project_folder(folder)
    length=40 if case in ('weir_recovered','bedload_stage_weir') else 20
    if length==40:
        # Extend only the downstream straight reach; retain groin, slope and dx.
        from osgeo import gdal
        original_path=folder/doc['terrain']['dem_path']; old=gdal.Open(str(original_path))
        projection=old.GetProjection();old=None
        path=folder.parent/'extended_channel.tif'
        raster=gdal.GetDriverByName('GTiff').Create(str(path),168,40,1,gdal.GDT_Float64)
        raster.SetGeoTransform((-1.,.25,0.,9.,0.,-.25));raster.SetProjection(projection)
        xs=-1+(np.arange(168)+.5)*.25
        raster.GetRasterBand(1).WriteArray(np.broadcast_to(.1-.001*xs,(40,168)).copy());raster=None
        doc['terrain']['dem_path']=str(path.resolve())
    doc = services.replace_domain(doc, [(0,0),(8,0),(8,3),(10,3),(10,0),(length,0),(length,8),(0,8)],
                                  'EPSG:2056', for_editing=True)
    inlet = case.removeprefix('dry_')
    if case == 'bedload_normal_critical': inlet = 'normal'
    if inlet not in ('total', 'stage', 'depth', 'critical', 'normal'): inlet = 'stage'
    if case == 'invalid_dry_total': inlet = 'total'
    im = 'inlet.' + dict(total='total_discharge', stage='stage_discharge', depth='depth_discharge',
                         critical='critical_discharge', normal='normal_discharge')[inlet]
    q = 16. if case in ('out_supercritical','supercritical_valid','free_supercritical') else 0. if case in ('zero','closed_weir') else 4.
    vals = dict(discharge_m3_s=q, stage_m=.6, depth_m=.5, datum_m=.1, energy_slope='mesh')
    inflow = dict(method=im, **{k: vals[k] for k in HYDRAULICS[im][2]})
    outlet = dict(method='outlet.stage', stage_m=.58)
    if case in ('out_critical','bedload_normal_critical','free_supercritical'): outlet = dict(method='outlet.critical')
    if case in ('out_weir','closed_weir','bedload_stage_weir','weir_recovered','weir_valid','bedload_weir_valid'):
        outlet = dict(method='outlet.free_weir', crest_m=.8 if case=='closed_weir' else .15,
                      coefficient_m_half_s=1.7, effective_width_m=8.)
    if case in ('out_supercritical','invalid_supercritical','supercritical_valid'): outlet = dict(method='outlet.supercritical')
    if case in ('backflow','closed_stage'): outlet['stage_m'] = .75
    for x, name, role, control in ((0,'Upstream','inlet',inflow),(length,'Downstream','outlet',outlet)):
        edges = PolygonDomain(**doc['settings']['domain']).edges
        ids = [e.edge_id for e in edges if e.start_xy[0] == e.end_xy[0] == x]
        doc = services.assign_boundary(doc, ids, name, role, control, for_editing=True)
    f = SettingsPanel(); f.load_document(doc)
    f.minimum_size.setValue(.5); f.levels.setValue(1); f.cfl.setValue(1.)
    set_choice(f.mesh_mode,'static_terrain'); set_choice(f.roughness,'roughness.chezy'); f.chezy.setValue(45.)
    f.end_time.setValue(end); f.dt_max.setValue(.1); f.output_interval.setValue(min(2., end/5))
    f.sediment_enabled.setChecked(sediment)
    if sediment:
        f.physics.bedload_on.setChecked(True); f.physics.grains.item(0,1).setText('1')
        f.physics.layers.item(0,0).setText('0'); f.sediment_start.setValue(0. if case=='bedload_weir_valid' else 40.); f.morfac.setValue(1.)
    doc = f.document(); f.close()
    doc['name'] = 'Groin boundary audit ' + case + ' ' + backend
    dry = case.startswith('dry_') or case == 'invalid_dry_total'
    doc['initial'].update(stage_m=-1. if dry else .6,
                          velocity_m_s=[4. if case in ('out_supercritical','supercritical_valid','free_supercritical') else 0., 0.])
    if case == 'backflow': doc['boundary_states']['Downstream']['stage_policy'] = 'stage.subcritical_backflow'
    if case == 'schedule':
        doc['boundary_states']['Upstream']['hydraulic_schedule'] = dict(method='forcing.linear_physical_time',
            time_s=[0.,10.,20.,30.,40.], discharge_m3_s=[4.,6.,0.,4.,4.],stage_m=[.6]*5)
    doc['boundary_resolution'] = resolve(doc['settings']['hydraulics'],doc['boundary_states'],
        dict(Upstream='inlet',Downstream='outlet'),sediment=doc['model']['physics']['sediment'],ng=1,end_time=end)
    doc.pop('editor_state',None)
    services.save_project_folder(folder,doc)
    return doc


def main():
    p=argparse.ArgumentParser();p.add_argument('--case', choices=CASES, required=True)
    p.add_argument('--backend', choices=('cpu','gpu'), required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    verify();install(json.loads((RUNTIME/'PRECOMPILED.json').read_text()),RUNTIME)
    app=QApplication([]);folder=root/'project';doc=document(folder,args.backend,args.case)
    from braids_desktop.preparation import prepare
    from braids_core import river_worker
    from braids_core.sediment.adaptive_river_worker import read_live_frame
    prepared=folder/'prepared/base';run=folder/'runs/audit';start=time.perf_counter()
    prepare(folder/'braids_project.json',prepared,asset_cache=root.parent/'asset_cache')
    owners=[];samples=[];original=river_worker.core.factory
    def factory(*a,**kw):
        make=original(*a,**kw)
        def build(*a,**kw):
            owner=make(*a,**kw);owners.append(owner)
            e=owner.exterior_bed;evaluate=e.evaluate
            def traced(q,z,**kwargs):
                try:return evaluate(q,z,**kwargs)
                except Exception as exc:
                    d=owner.data;mask=(d['neighbour']<0)&(d['bid']>=0);ids=d['owner'][mask]
                    state=host(q);hh=state[0,ids];normal=d['normal'][:,mask]
                    un=np.divide(np.sum(state[1:,ids]*normal,axis=0),hh,out=np.zeros_like(hh),where=hh>1e-6)
                    record=dict(time_s=float(kwargs.get('time',owner.time)),error=str(exc),
                        face_status=host(e.status)[mask].tolist(),boundary_id=d['bid'][mask].tolist(),
                        xy=d['mid'][:,mask].T.tolist(),depth_m=hh.tolist(),normal_velocity_m_s=un.tolist(),
                        Froude=np.divide(abs(un),np.sqrt(9.81*hh),out=np.zeros_like(hh),where=hh>1e-6).tolist())
                    if e.free_weir is not None:record['weir_status']=host(e.free_weir.status).tolist()
                    (root/'REJECTION.json').write_text(json.dumps(record,indent=2));raise
            e.evaluate=traced
            return owner
        build.__dict__.update(make.__dict__)
        return build
    river_worker.core.factory=factory
    def sample(folder,row):
        a=owners[-1];e=a.exterior_bed;d=a.data
        before=host(a.q).copy();zbefore=host(a.z).copy()
        e.evaluate(a.q,a.z,time=float(row['time_s']))
        assert np.array_equal(before,host(a.q)) and np.array_equal(zbefore,host(a.z))
        trace=host(e.trace);flux=host(e.flux);bed=host(e.p.exterior_bed);forcing=host(e.forcing)
        item=dict(time_s=float(row['time_s']), minimum_depth_m=float(before[0].min()),boundaries={})
        # Prepared case file owns the mapping; do not infer IDs from dict order.
        from braids_desktop.results import ResultsStore
        store=ResultsStore(run);frame=store.read(-1)
        for name in ('Upstream','Downstream'):
            mask=store._boundary_mask(frame,name);b=int(d['bid'][mask][0]);L=d['length'][mask]
            h=trace[0,mask];un=np.divide(np.sum(trace[1:,mask]*d['normal'][:,mask],axis=0),h,out=np.zeros_like(h),where=h>1e-6)
            value=dict(Q_out_m3_s=float(np.dot(flux[0,mask],L)),
                trace_stage_min_m=float(np.min(h+bed[mask])),trace_stage_max_m=float(np.max(h+bed[mask])),
                reference_stage_m=float(forcing[1,b]),minimum_local_Q_m2_s=float(flux[0,mask].min()),
                maximum_local_Q_m2_s=float(flux[0,mask].max()),
                trace_Froude_min=float(np.min(np.divide(abs(un),np.sqrt(9.81*h),out=np.zeros_like(h),where=h>1e-6))))
            if e.section is not None and name=='Upstream':
                href=np.maximum(forcing[1,b]-bed[mask],0);A=np.dot(href,L);B=L[href>1e-6].sum()
                value['reference_section_Fr2']=float(forcing[0,b]**2*B/(9.81*A**3)) if A else 0.
            if e.normal_section is not None and name=='Upstream':value['normal_metrics']=host(e.normal_section.metrics())[:,b].tolist()
            if e.free_weir is not None and name=='Downstream':
                value['weir_stage_m']=float(host(e.free_weir.stage)[b])
                value['weir_branch']=host(e.free_weir.branch)[mask].tolist()
            if e.critical_outlet is not None and name=='Downstream':
                active=host(e.critical_outlet.branch)[mask]==1
                A=np.dot(h[active],L[active]);B=L[active].sum();Q=np.dot(flux[0,mask][active],L[active])
                value['controlled_section_Fr2']=float(Q**2*B/(9.81*A**3)) if A else None
            item['boundaries'][name]=value
        samples.append(item)
        (root/'TRACES.json').write_text(json.dumps(samples,indent=2))
    expected=args.case.startswith('invalid_');error=None
    try:status=river_worker.run(prepared,run,on_frame=sample,checkpoint_dir=folder/'checkpoints/audit')
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
        if not expected:raise
        status=dict(status='expected_rejection')
    finally:river_worker.core.factory=original
    if expected:
        assert error is not None,'Inapplicable hydraulic configuration was accepted'
        assert 'Exterior hydraulic boundary rejected' in error,error
    else:
        assert status['status']=='complete',status
        rows=json.loads((run/'DIAGNOSTICS.json').read_text())
        assert max(abs(r['water_residual_m3']) for r in rows)<1e-8
        assert all(not any(r['wall_integrals']) for r in rows)
        for r in samples:
            target=doc['settings']['hydraulics']['Upstream']['discharge_m3_s']
            if args.case=='schedule':
                table=doc['boundary_states']['Upstream']['hydraulic_schedule'];target=np.interp(r['time_s'],table['time_s'],table['discharge_m3_s'])
            assert abs(-r['boundaries']['Upstream']['Q_out_m3_s']-target)<1e-9,r
            assert r['minimum_depth_m']>=0
    result=dict(case=args.case,backend=args.backend,status=status['status'],error=error,
                elapsed_s=time.perf_counter()-start,
                runtime_manifest_sha256=hashlib.sha256((RUNTIME/'RUNTIME.json').read_bytes()).hexdigest())
    (root/'CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
