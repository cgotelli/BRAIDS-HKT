"""Short linear-wave boundary diagnostic using unchanged installed engines.

The preparation callback supplies an identified analytic initial condition only;
it does not change reconstruction, fluxes, boundaries, stepping or restart files.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'desktop/runtime'), str(ROOT), str(ROOT/'desktop'),
               str(ROOT/'validation/sediment/delft3d_current')]
from check_jet173 import document, setup
from stage_wall172 import save, sha

BASE = ROOT/'validation/_pulse177/matched'
H, U, G, WIDTH, LENGTH, DX = .6, 4./(8*.6), 9.81, 8., 40., .5
C = np.sqrt(G*H)
AMP, SIGMA, END, OUTPUT, CHEZY = .001, 1.5, 20., .1, 1000.


def wave(x, case, average=False):
    x = np.asarray(x, dtype=float)
    if case == 'base':
        return np.zeros_like(x)
    center = 25. if case == 'right' else 15.
    if average:
        return AMP*SIGMA*np.sqrt(np.pi/2)/DX*(
            erf((x+DX/2-center)/(np.sqrt(2)*SIGMA))-
            erf((x-DX/2-center)/(np.sqrt(2)*SIGMA)))
    return AMP*np.exp(-.5*((x-center)/SIGMA)**2)


def initial(x, case):
    # Cell averages of h and hu, not the product of independently averaged h,u.
    x=np.asarray(x);nodes,weights=np.polynomial.legendre.leggauss(16)
    a=wave(x[:,None]+DX/2*nodes,case)
    direction=1 if case=='right' else -1
    h=H+a;u=U+direction*C/H*a
    return np.vstack([h@weights/2,(h*u)@weights/2,np.zeros(len(x))])


def definition(case):
    return dict(case=case,depth_m=H,velocity_m_s=U,discharge_m3_s=4.,gravity=G,
        length_m=LENGTH,width_m=WIDTH,cell_m=DX,amplitude_m=AMP,sigma_m=SIGMA,
        center_m=None if case=='base' else 25. if case=='right' else 15.,
        end_s=END,output_s=OUTPUT,chezy=CHEZY,physical_viscosity=0.,
        inlet='fixed total discharge with reference stage',outlet='fixed stage, backflow allowed',
        expected_stage_surface_reflection=-1.,expected_discharge_surface_reflection=(C-U)/(C+U),
        caveat='Flat, weak-drag diagnostic, not the sloping rough groin. '
               'The matching no-pulse control is subtracted before characteristic analysis. '
               'Same boundary types. Sediment inactive throughout. No engine changes.')


def braids(case,backend):
    import rasterio
    from rasterio.transform import from_origin
    from braids_core import river_worker
    from braids_core.river_open_boundary import resolve
    setup();out=BASE/case/backend;out.mkdir(parents=True,exist_ok=False)
    dem=out/'flat.tif'
    with rasterio.open(dem,'w',driver='GTiff',width=84,height=20,count=1,dtype='float64',
                       crs='EPSG:2056',transform=from_origin(-1,9,.5,.5)) as f:
        f.write(np.zeros((20,84)),1)
    d=document(backend,END)
    d['terrain']['dem_path']=str(dem.resolve())
    d['settings']['domain']['vertices_xy']=[[0.,0.],[40.,0.],[40.,8.],[0.,8.],[0.,0.]]
    d['settings']['output_interval_s']=OUTPUT
    d['settings']['controls']['chezy']=CHEZY
    d['roughness_parameters']['chezy_m_half_s']=CHEZY
    d['settings']['hydraulics']['Downstream']['stage_m']=H
    d['initial'].update(stage_m=H,velocity_m_s=[U,0.])
    d['boundary_resolution']=resolve(d['settings']['hydraulics'],d['boundary_states'],
        dict(Upstream='inlet',Downstream='outlet'),sediment=d['model']['physics']['sediment'],ng=1,end_time=END)
    save(out/'project.json',d);save(out/'INITIAL_DEFINITION.json',definition(case))
    original=river_worker.prepare_recipe
    def recipe(*args,**kw):
        callback=kw.get('on_fields')
        def fields(tree,a):
            if callback is not None:callback(tree,a)
            xy=np.array([cell.centroid_xy for cell in tree.finest.cells])
            a['q'][:]=initial(xy[:,0],case)
            np.savez(out/'INITIAL.npz',xy=xy,q=a['q'],z=a['rock']+a['thickness'].sum(0))
        kw['on_fields']=fields
        return original(*args,**kw)
    river_worker.prepare_recipe=recipe
    prepared=river_worker.prepare(out/'project.json',out/'prepared')
    status=river_worker.run(prepared,out/'run')
    save(out/'CHECK.json',dict(status=status,definition=definition(case),
        initial_sha256=sha(out/'INITIAL.npz'),script_sha256=sha(__file__),
        runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),production_modified=False))
    print(json.dumps(status),flush=True)


def native(case, datum=0., chezy=CHEZY, label=None):
    import runtime
    from check_bedload161 import replace,table
    parent=ROOT/'validation/_corner174/delft';out=BASE/case/(label or ('delft' if datum==0 else 'delft_datum'))
    out.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((parent/'INPUTS.json').read_text())
    for name,digest in receipt['inputs'].items():
        assert sha(parent/name)==digest,name
        shutil.copy2(parent/name,out/name)
    changes=dict(Tstop=str(END),Flmap=f'0 {OUTPUT} {END}',Flhis=f'0 {OUTPUT} {END}',
                 Ccofu=str(chezy),Ccofv=str(chezy),FilDry='##')
    code=(out/'sim_P057.mdf').read_text()
    for k,v in changes.items():code=replace(code,k,v)
    (out/'sim_P057.mdf').write_text(code)
    (out/'mor.mor').write_text(replace((out/'mor.mor').read_text(),'MorStt',str(END+1)))
    xc=(np.arange(82)-.5)*DX;xf=np.arange(82)*DX
    # Stored array index j maps to x=(j-.5)dx for scalar and x=j*dx for U1.
    # The actual serialized grid/initial output is checked before comparison.
    bed=np.full((18,82),-datum);eta=np.broadcast_to(H+datum+wave(xc,case,True),bed.shape).copy()
    sign=1 if case=='right' else -1
    velocity=np.broadcast_to(U+sign*C/H*wave(xf,case),bed.shape).copy()
    np.savetxt(out/'bed.dep',bed,fmt='%.17e')
    np.savetxt(out/'initial.ini',np.concatenate([eta,velocity,np.zeros_like(bed)]),fmt='%.17e')
    (out/'flume.bct').write_text(table('upstream','total discharge (t)','m3/s',4.,True)+
                               table('downstream','water elevation (z)','m',H+datum,True))
    save(out/'INITIAL_DEFINITION.json',definition(case))
    save(out/'INPUTS.json',dict(parent=str(parent),parent_manifest_sha256=sha(parent/'INPUTS.json'),
        changes=dict(mdf=changes,initial=definition(case),bed_datum_m=datum,MorStt=END+1),
        Delft3D_source_modified=False,Delft3D_runtime_modified=False,
        inputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(runtime.run_case(out,configuration='config_flow2d3d.xml',timeout_s=180)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('backend',choices=('cpu','gpu','delft'))
    ap.add_argument('case',choices=('base','left','right'));a=ap.parse_args()
    if a.backend=='delft':native(a.case)
    else:braids(a.case,a.backend)
