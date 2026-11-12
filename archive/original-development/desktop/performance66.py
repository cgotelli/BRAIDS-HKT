"""Isolated high-TI cold-start coupled validation. Installed Desktop untouched."""
import argparse,hashlib,json,os,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'desktop/qualification/coupled_inflow65/candidate'
CANDIDATE=ROOT/'desktop/qualification/performance66/candidate'
REPORT=ROOT/'validation/hydrokinetic_turbines/reports/performance66'
HOME=ROOT/'desktop/qualification/performance66/runs'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def stage():
    CANDIDATE.mkdir(parents=True,exist_ok=False)
    old=read(BASE/'RUNTIME.json');files=dict(old['files']);edit='braids_core/sediment/coupled_inflow65.py';a=read(REPORT/'AUDIT.json')
    for n,h in files.items():
        s=BASE/n;t=CANDIDATE/n;t.parent.mkdir(parents=True,exist_ok=True);assert sha(s)==h,n
        if n==edit:shutil.copy2(s,t)
        else:os.link(s,t)
    source=(CANDIDATE/edit).read_text(encoding='utf-8')
    source=source.replace('GROWTH=.017872630937080202',f'GROWTH={a["growth"]!r}').replace('INITIAL_WIDTH=.2806331261614108',f'INITIAL_WIDTH={a["initial_width"]!r}')
    source=source.replace('research.coupled_gaussian_rotor_moments.v1','research.coupled_gaussian_rotor_moments.ti15.v1')
    source=source.replace('Mycek TI3%, stations4D/8D; milestone64','Mycek TI15%, stations4D/8D; milestone66')
    # A single rotor has no incoming upstream wake; use the unchanged source.
    source=source.replace("if len(rows)!=2:raise ValueError('Research65 requires exactly two rotors')", "self.single = len(rows)==1\n            if self.single:return\n            if len(rows)!=2:raise ValueError('Research66 supports one or two rotors')")
    source=source.replace('up_index,down_index=self.order;', 'if self.single:return [r.sample(q,z,grad) for r in self.rotors]\n            up_index,down_index=self.order;')
    (CANDIDATE/edit).write_text(source,encoding='utf-8');files[edit]=sha(CANDIDATE/edit)
    save(CANDIDATE/'RUNTIME.json',dict(old,files=files));save(REPORT/'STAGE.json',dict(parent=sha(BASE/'RUNTIME.json'),candidate=sha(CANDIDATE/'RUNTIME.json'),installed=sha(ROOT/'desktop/runtime/RUNTIME.json'),native_unchanged=True,calibration=a))
def run(spacing,backend,nu_scale=1.):
    sys.path[:0]=[str(CANDIDATE),str(ROOT)];sys.dont_write_bytecode=True;os.environ['QT_QPA_PLATFORM']='offscreen'
    if os.name=='nt':dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
    from braids_core.sediment import portable_river_runtime
    from braids_core import river_worker
    from braids_core.river_model import resolve_model
    from braids_desktop import services,preparation
    from validation.hydrokinetic_turbines.performance66 import extract,coefficient
    import numpy as np
    portable_river_runtime.install(read(CANDIDATE/'PRECOMPILED.json'),CANDIDATE)
    folder=HOME/f'{spacing}D_{backend}_nu{nu_scale:g}';folder.mkdir(parents=True,exist_ok=False)
    doc=read(ROOT/f'desktop/qualification/coupled_inflow65/runs/{10 if spacing==10 else 6}D_cpu_research/input.json')
    doc['model']['request']['execution']['backend']=backend;doc['model']=resolve_model(doc['model']['request']).document()
    doc['initial']['velocity_m_s']=[.83,0.];doc['settings'].update(end_time_s=60.,output_interval_s=1.,checkpoint_interval_s=60.)
    doc['settings']['controls']['threads']=8
    doc['settings']['hydraulics']['Upstream']['discharge_m3_s']=.83*8
    rows=extract();performance=[dict(tip_speed_ratio=float(t),thrust_coefficient=coefficient(rows,15,.83,t,'Ct'),power_coefficient=coefficient(rows,15,.83,t,'Cp')) for t in np.arange(1.,8.26,.25)]
    turbines=doc['turbines']['turbines']
    if spacing==0:doc['turbines']['turbines']=turbines[:1]
    # Retain the qualified coordinates bit-for-bit (including binary rounding).
    for i,r in enumerate(doc['turbines']['turbines']):
        r['control'].update(model_name='Mycek Part I TI15 U0.83 measured curve',performance=performance,rotational_speed_rpm=(3.67 if i==0 else 4)*.83/(.7/2)*30/np.pi)
    doc['turbines']['ambient_viscosity_m2_s']=read(REPORT/'AUDIT.json')['ambient_viscosity_m2_s']*nu_scale
    doc=services.validate_document(doc);save(folder/'input.json',doc)
    prepared=preparation.prepare(folder/'input.json',folder/'prepared',asset_cache=HOME/'assets')
    save(folder/'SPEC.json',dict(runtime=sha(CANDIDATE/'RUNTIME.json'),spacing_D=spacing,backend=backend,nu_scale=nu_scale,start='cold uniform 0.83m/s, depth2m',nu=doc['turbines']['ambient_viscosity_m2_s']))
    start=time.monotonic();print('Running',folder.name,flush=True)
    result=river_worker.run(prepared,folder/'run')
    save(folder/'DONE.json',dict(result=result,wall_seconds=time.monotonic()-start));print('Complete',folder.name,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','run']);p.add_argument('--spacing',type=int,choices=[0,6,10],default=0);p.add_argument('--backend',choices=['cpu','gpu'],default='cpu');p.add_argument('--nu-scale',type=float,default=1.);a=p.parse_args()
    stage() if a.action=='stage' else run(a.spacing,a.backend,a.nu_scale)
