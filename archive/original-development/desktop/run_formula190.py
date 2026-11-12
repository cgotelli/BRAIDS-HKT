"""Public Desktop preparation, coupled groin, feed and restart for new laws."""
import argparse,json,os,sys,time
from pathlib import Path
from stage_formula190 import ROOT,TARGET,OUT
from stage_secondary185 import save,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=('cpu','gpu'),required=True)
    ap.add_argument('--law',choices=('ancey_recking_2023','recking_2010','camenen_larson_2005','wong_parker_2006'),required=True)
    ap.add_argument('--dry-amr',action='store_true');args=ap.parse_args()
    os.environ['QT_QPA_PLATFORM']='offscreen';os.environ['OMP_NUM_THREADS']='4'
    os.environ['BRAIDS_AUDIT_RUNTIME']=str(TARGET)
    sys.path[:0]=[str(TARGET),str(ROOT),str(ROOT/'tests'),str(ROOT/'desktop')]
    import numpy as np
    from PySide6.QtWidgets import QApplication
    from osgeo import gdal
    from check_boundaries159 import document
    from braids_desktop.forms import SettingsPanel
    from braids_desktop.river_physics import set_choice
    from braids_desktop import services
    from braids_desktop.preparation import prepare
    from braids_desktop.results import ResultsStore
    from braids_core import river_worker
    from braids_core.river_launcher import verify
    from braids_core.river_open_boundary import resolve
    from braids_core.sediment.portable_river_runtime import install
    from braids_core.sediment.bed_columns_v101 import host
    verify();install(json.loads((TARGET/'PRECOMPILED.json').read_text()),TARGET)
    import subprocess
    original_popen=subprocess.Popen
    def guarded(command,*a,**kw):
        executable=str(command[0] if isinstance(command,(list,tuple)) else command).lower()
        if any(x in executable for x in ('gfortran','g++','gcc','clang')):
            raise AssertionError('Portable run requested a compiler: '+executable)
        return original_popen(command,*a,**kw)
    subprocess.Popen=guarded
    app=QApplication.instance() or QApplication([])
    out=OUT/'runs'/('dry' if args.dry_amr else 'wet')/(args.law+'_'+args.backend)
    out.mkdir(parents=True,exist_ok=False)
    doc=document(out/'project',args.backend,'stage');path=Path(doc['terrain']['dem_path'])
    if not path.is_absolute():path=out/'project'/path
    newpath=out/'channel.tif';gdal.Translate(str(newpath),str(path))
    doc['terrain']['dem_path']=str(newpath.resolve())
    raster=gdal.Open(str(newpath),gdal.GA_Update);gt=raster.GetGeoTransform()
    x=gt[0]+(np.arange(raster.RasterXSize)+.5)*gt[1]
    raster.GetRasterBand(1).WriteArray(np.broadcast_to(.1-.01*x,(raster.RasterYSize,len(x))).copy());raster=None
    f=SettingsPanel();f.load_document(doc);f.sediment_enabled.setChecked(True);f.physics.bedload_on.setChecked(True)
    set_choice(f.physics.capacity,'capacity.'+args.law)
    f.physics.grains.item(0,1).setText('5');f.physics.layers.item(0,0).setText('.1')
    if args.dry_amr:f.levels.setValue(2);set_choice(f.mesh_mode,'dynamic_depth')
    f.chezy.setValue(20.);f.end_time.setValue(16.);f.dt_max.setValue(.05);f.cfl.setValue(.8)
    f.output_interval.setValue(2.);f.sediment_start.setValue(4.);f.morfac.setValue(1.)
    doc=f.document();f.close();doc.pop('editor_state',None)
    doc['name']=args.law+' groin '+args.backend
    doc['settings']['hydraulics']['Upstream'].update(discharge_m3_s=2.,stage_m=.4)
    doc['settings']['hydraulics']['Downstream'].update(stage_m=.2)
    doc['initial'].update(stage_m=-1. if args.dry_amr else .4,velocity_m_s=[0.,0.])
    doc['settings']['adaptation']['domain_border_enabled']=False
    doc['settings']['checkpoint_interval_s']=8.
    doc['boundary_resolution']=resolve(doc['settings']['hydraulics'],doc['boundary_states'],
        dict(Upstream='inlet',Downstream='outlet'),sediment=doc['model']['physics']['sediment'],ng=1,end_time=16.)
    services.save_project_folder(out/'project',doc)
    owners=[];original=river_worker.core.factory
    def traced(*a,**kw):
        maker=original(*a,**kw)
        def build(*aa,**kk):
            owner=maker(*aa,**kk);owners.append(owner);return owner
        build.__dict__.update(maker.__dict__);return build
    river_worker.core.factory=traced
    prep=out/'prepared';prepare(out/'project/braids_project.json',prep,asset_cache=OUT/'assets')
    began=time.perf_counter();status=river_worker.run(prep,out/'full',checkpoint_dir=out/'checkpoints')
    elapsed=time.perf_counter()-began;assert status['status']=='complete',status
    owner=owners[-1];save(out/'OWNER.json',owner.identity)
    frame=ResultsStore(out/'full').read(-1);xy=np.array([p.centroid.coords[0] for p in frame.polygons])
    dz=frame.z-(.1-.01*xy[:,0])
    assert np.isfinite(frame.q).all() and frame.q[0].min()>=0 and frame.arrays['mass'].min()>=0
    assert dz.max()>1e-8 and dz.min()<-1e-8
    diag=json.loads((out/'full/DIAGNOSTICS.json').read_text())
    water=max(abs(d['water_residual_m3']) for d in diag)
    sediment=max(np.max(abs(np.asarray(d['class_residual_kg']))) for d in diag)
    assert water<1e-8 and sediment<1e-7
    assert all(not any(d['wall_integrals']) for d in diag)
    ext=owner.physics.exterior_sediment;q=host(ext.trace);h=q[0]
    speed=np.divide(np.hypot(*q[1:]),h,out=np.zeros_like(h),where=h>0)
    c=host(ext.capacity[0].hydraulic_chezy);theta=speed**2/c**2/(1.65*.005)
    from test_ancey_recking190 import phi
    from test_recking190 import oracle
    from test_camenen192 import phi as camenen_phi
    dimensionless=(3.97*np.maximum(theta-.0495,0)**1.5 if args.law=='wong_parker_2006' else
                   camenen_phi(theta) if args.law=='camenen_larson_2005' else
                   phi(theta) if args.law=='ancey_recking_2023' else oracle(theta,.01))
    expected=dimensionless*np.sqrt(9.81*1.65*.005**3)
    actual=np.linalg.norm(host(ext.values[0])[:2],axis=0)/2650.
    wet=h>owner.physics.sediment_depth
    np.testing.assert_allclose(actual[wet],expected[wet],rtol=3e-12,atol=1e-16)
    stop=out/'STOP'
    def cancel(folder,row):
        if row['time_s']>=8.:stop.touch()
    part=river_worker.run(prep,out/'cancelled',stop_file=stop,on_frame=cancel)
    assert part['status']=='cancelled' and part['time_s']==8.,part
    resumed=river_worker.run(prep,out/'resumed',restart=out/'cancelled/checkpoint.npz')
    assert resumed['status']=='complete',resumed
    final=ResultsStore(out/'resumed').read(-1)
    exact={k:bool(np.array_equal(frame.arrays[k],final.arrays[k])) for k in ('q','z','mass','water','bed_boundary')}
    assert all(exact.values()),exact
    save(out/'CHECK.json',dict(status=status,runtime=sha(TARGET/'RUNTIME.json'),elapsed_s=elapsed,
        bed_min_m=float(dz.min()),bed_max_m=float(dz.max()),water_residual_m3=water,
        sediment_residual_kg=float(sediment),restart_exact=exact,compiler_free=True,inlet_capacity_oracle=True))
    print(args.backend,args.law,'dry' if args.dry_amr else 'wet','PASS',elapsed,flush=True)
if __name__=='__main__':main()
