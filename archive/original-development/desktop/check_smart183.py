"""Short compiler-free public groin run and exact checkpoint continuation."""
import argparse,json,os,sys,time
from pathlib import Path
from stage_smart183 import ROOT,TARGET,OUT,sha,save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=('cpu','gpu'),required=True)
    ap.add_argument('--law',default='smart_jaggi',choices=('smart_jaggi','mpm','ashida_michiue_generalized','wang_fredsoe'))
    ap.add_argument('--runtime',type=Path,default=TARGET);ap.add_argument('--tag',default='v1')
    ap.add_argument('--dry-amr',action='store_true');ap.add_argument('--stock',type=float,default=.1)
    ap.add_argument('--secondary',action='store_true',help='Explicit additional 185 regression, not the historical 183 input')
    args=ap.parse_args();runtime=args.runtime.resolve()
    os.environ['QT_QPA_PLATFORM']='offscreen';os.environ['BRAIDS_AUDIT_RUNTIME']=str(runtime)
    sys.path[:0]=[str(runtime),str(ROOT),str(ROOT/'desktop')]
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
    verify();install(json.loads((runtime/'PRECOMPILED.json').read_text()),runtime)
    import subprocess
    popen=subprocess.Popen
    def guarded(command,*a,**kw):
        executable=str(command[0] if isinstance(command,(list,tuple)) else command).lower()
        if any(x in executable for x in ('gfortran','g++','gcc','clang')):
            raise AssertionError('Portable run requested a compiler: '+executable)
        return popen(command,*a,**kw)
    subprocess.Popen=guarded
    app=QApplication.instance() or QApplication([])
    out=OUT/args.tag/(args.law+'_'+args.backend);out.mkdir(parents=True,exist_ok=False)
    doc=document(out/'project',args.backend,'stage')
    path=Path(doc['terrain']['dem_path']);path=path if path.is_absolute() else out/'project'/path
    # Project inputs are content-addressed; create a new input, never mutate one.
    newpath=out/'steep_channel.tif';gdal.Translate(str(newpath),str(path))
    doc['terrain']['dem_path']=str(newpath.resolve())
    raster=gdal.Open(str(newpath),gdal.GA_Update);gt=raster.GetGeoTransform()
    x=gt[0]+(np.arange(raster.RasterXSize)+.5)*gt[1]
    raster.GetRasterBand(1).WriteArray(np.broadcast_to(.1-.01*x,(raster.RasterYSize,len(x))).copy());raster=None
    f=SettingsPanel();f.load_document(doc)
    f.sediment_enabled.setChecked(True);f.physics.bedload_on.setChecked(True)
    set_choice(f.physics.capacity,'capacity.'+args.law)
    f.physics.grains.item(0,1).setText('5');f.physics.layers.item(0,0).setText(str(args.stock))
    if args.dry_amr:
        f.levels.setValue(2);set_choice(f.mesh_mode,'dynamic_depth')
    if args.secondary:
        f.secondary.setChecked(True);f.betac.setValue(1.)
        f.secondary_diffusion.setValue(.01);f.turbulence.setChecked(True);f.viscosity.setValue(.01)
    f.chezy.setValue(20.);f.end_time.setValue(16.);f.dt_max.setValue(.05);f.cfl.setValue(.8)
    f.output_interval.setValue(2.);f.sediment_start.setValue(4.);f.morfac.setValue(1.)
    doc=f.document();f.close();doc.pop('editor_state',None)
    doc['name']='Smart-Jaeggi steep groin verification '+args.backend
    doc['settings']['hydraulics']['Upstream'].update(discharge_m3_s=2.,stage_m=.4)
    doc['settings']['hydraulics']['Downstream'].update(stage_m=.2)
    doc['initial'].update(stage_m=.4,velocity_m_s=[0.,0.])
    if args.dry_amr:
        doc['initial']['stage_m']=-1.
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
    t=time.perf_counter();status=river_worker.run(prep,out/'full',checkpoint_dir=out/'checkpoints');elapsed=time.perf_counter()-t
    assert status['status']=='complete',status
    owner=owners[-1];save(out/'OWNER.json',owner.identity)
    first=ResultsStore(out/'full').read(0);last=ResultsStore(out/'full').read(-1)
    assert np.isfinite(last.arrays['q']).all() and last.arrays['q'][0].min()>=0
    assert last.arrays['mass'].min()>=0
    dz=None if args.dry_amr else last.arrays['z']-first.arrays['z']
    # A moving mesh has different cell order/count: use prepared analytic bed.
    if args.dry_amr:
        xy=np.array([list(p.centroid.coords)[0] for p in last.polygons]);dz=last.arrays['z']-(.1-.01*xy[:,0])
    assert dz.max()>1e-8, (dz.min(),dz.max())
    if args.stock:assert dz.min()<-1e-8
    else:assert dz.min()>-1e-9
    diag=json.loads((out/'full/DIAGNOSTICS.json').read_text())
    water=max(abs(d['water_residual_m3']) for d in diag)
    sediment=max(np.max(abs(np.asarray(d['class_residual_kg']))) for d in diag)
    assert water<1e-8 and sediment<1e-7,(water,sediment)
    assert all(not any(d['wall_integrals']) for d in diag)
    restart={}
    if args.law=='smart_jaggi':
        from braids_core.sediment.river_smart_jaggi import __file__ as smart_source
        ext=owner.physics.exterior_sediment
        qext=host(ext.trace);h=qext[0];speed=np.divide(np.hypot(*qext[1:]),h,out=np.zeros_like(h),where=h>0)
        c=host(ext.capacity[0].hydraulic_chezy);delta=2650./1000.-1.;j=speed**2/c**2/np.maximum(h,1e-30)
        expected=4./delta*j**.6*h*speed*np.maximum(j-.05*delta*.005/np.maximum(h,1e-30),0.)
        actual=np.linalg.norm(host(ext.values[0])[:2],axis=0)/2650.
        wet=h>owner.physics.sediment_depth
        np.testing.assert_allclose(actual[wet],expected[wet],rtol=2e-12,atol=1e-16)
        # Independently stopped prefix then restart, with no reinterpreted identity.
        stop=out/'STOP'
        def cancel(folder,row):
            if row['time_s']>=8.:stop.touch()
        part=river_worker.run(prep,out/'cancelled',stop_file=stop,on_frame=cancel)
        assert part['status']=='cancelled' and part['time_s']==8.,part
        resumed=river_worker.run(prep,out/'resumed',restart=out/'cancelled/checkpoint.npz')
        assert resumed['status']=='complete'
        result=ResultsStore(out/'resumed').read(-1)
        exact={k:bool(np.array_equal(last.arrays[k],result.arrays[k])) for k in ('q','z','mass','water','bed_boundary')}
        assert all(exact.values()),exact
        restart=dict(status=resumed,exact=exact)
        # The inlet's capacity feed must consume this same new formula.
        assert all(c.identity['method']=='capacity.smart_jaggi' for c in owner.capacity)
    save(out/'CHECK.json',dict(status=status,runtime=sha(runtime/'RUNTIME.json'),elapsed_s=elapsed,
         bed_min_m=float(dz.min()),bed_max_m=float(dz.max()),water_residual_m3=water,
         sediment_residual_kg=float(sediment),restart=restart,compiler_free=True,
         inlet_capacity_oracle=args.law=='smart_jaggi',dry_amr=args.dry_amr,initial_stock_m=args.stock))
    print(args.backend,args.law,'PASS',float(dz.min()),float(dz.max()),elapsed,flush=True)
if __name__=='__main__':main()
