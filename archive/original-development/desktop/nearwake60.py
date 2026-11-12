"""Near-wake diagnosis: fixed loading and mixing; explicit mesh controls."""
import argparse,copy,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'desktop/runtime'
HOME=Path.home()/'Desktop/BRAIDS_Turbine_NearWake60'
REPORT=ROOT/'validation/hydrokinetic_turbines/reports/nearwake60'
PARENT=ROOT/'validation/hydrokinetic_turbines/reports/desktop_validation59'
sys.path.insert(0,str(RUNTIME));sys.path.insert(1,str(ROOT/'desktop'))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def create(name):
    from braids_desktop import services,project_file
    oldhome=Path.home()/'Desktop/BRAIDS_Turbine_Validation59'
    records=read(oldhome/'PROJECTS.json');r=records['nash_gpu']
    _,doc,_,_=project_file.read(Path(r['project']))
    size=1.25 if name.startswith(('wake8','zero8')) else 2.5
    c=copy.deepcopy(r['configuration']);c.update(folder=name,dx=size,dy=size)
    doc['name']=name;doc['settings']['levels']=3 if size==1.25 else 2
    # Identical 5m coarse grid and exact physical geometry, with 2.5m finest.
    doc['settings']['breaklines']=[]
    if name in ('wake4_gpu','wake8_gpu','wake4_cpu','zero4_gpu','zero8_gpu','zero8_corrected'):
        doc['settings']['breaklines']=[dict(id=f'wake_refinement_{i}',closed=False,
            points_xy=[[480.,270.+i*size],[750.,270.+i*size]]) for i in range(round(50/size)+1)]
    elif name!='rotor4_gpu':raise ValueError(name)
    if name.startswith('zero'):
        # Same rotor object, footprint pins and wake mesh; only physical loading
        # is zero. A bounded control must pass stationarity before normalization.
        doc['turbines']['turbines'][0]['control']['thrust_coefficient']=0.
        doc['settings']['end_time_s']=500.;doc['settings']['checkpoint_interval_s']=500.
        doc.setdefault('editor_state',{})['checkpoint_clock']=dict(mode='end_time',custom_interval_s=500.)
        c.update(end=500.,average=400.,ct=0.)
    if name.endswith('_cpu'):
        from braids_core.river_model import resolve_model
        doc['model']['request']['execution']['backend']='cpu';doc['model']=resolve_model(doc['model']['request']).document()
        doc['settings']['end_time_s']=100.;doc['settings']['checkpoint_interval_s']=100.
        doc.setdefault('editor_state',{})['checkpoint_clock']=dict(mode='end_time',custom_interval_s=100.)
        c.update(backend='cpu',end=100.,average=50.)
    doc['settings']['adaptation']['transition_layers']=2
    doc=services.validate_document(doc)
    HOME.mkdir(exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    folder=HOME/name;folder.mkdir(exist_ok=False)
    oldroot=Path(r['folder'])/'Project_data'
    old_session=services.project_session(oldroot)
    saved=services.save_project_folder(folder/'Project_data',doc,session=dict(desktop=copy.deepcopy(old_session['desktop'])))
    project_file.write(folder/'Validation.braids.json',folder/'Project_data',saved)
    manifest=read(HOME/'PROJECTS.json') if (HOME/'PROJECTS.json').exists() else {}
    runtime=ROOT/'desktop/qualification/nearwake60/candidate' if name.endswith('_corrected') else RUNTIME
    manifest[name]=dict(project=str(folder/'Validation.braids.json'),folder=str(folder),configuration=c,runtime_sha256=sha(runtime/'RUNTIME.json'))
    save(HOME/'PROJECTS.json',manifest)
    save(folder/'AUDIT_SPEC.json',dict(runtime=sha(runtime/'RUNTIME.json'),parent=str(PARENT),
        purpose='Isolate rotor-only versus rotor-and-wake resolution. Loaded GPU cases retain original physics and duration; zero-load and bounded CPU controls are explicitly identified.',
        refinement='Mesh-only axial breaklines in the flat fixed-bed domain; not solid barriers.' if doc['settings']['breaklines'] else 'Native rotor footprint pins only.',
        end_time_s=c['end'],average_start_s=c['average'],configuration=c))
    save(REPORT/'LOCKED_SPEC.json',dict(runtime_sha256=sha(RUNTIME/'RUNTIME.json'),projects=manifest))
    print(project_file.__file__,folder,flush=True)

def run(name):
    import run_nearwake60_ui as ui
    ui.RUNTIME=RUNTIME;ui.HOME=HOME;ui.REPORT=REPORT
    ui.run(name)

def stop(name):
    row=read(REPORT/name/'STARTED.json');output=Path(row['output']).resolve()
    assert output.is_relative_to(HOME.resolve())
    progress=read(output/'PROGRESS.json');assert progress['time_s']>=600
    header=json.loads((output.parent.parent/'history'/(output.name+'.log')).open(encoding='utf-8').readline())
    args=header['arguments'];flag=Path(args[args.index('--stop-file')+1]).resolve()
    assert flag.parent==output.parent and flag.name.startswith('.braids_stop_')
    flag.touch(exist_ok=False)
    save(REPORT/name/'BOUNDED_STOP.json',dict(output=str(output),time_at_request=progress['time_s'],
        reason='Rotor-only diagnostic: near wake retained through 600s; full far-wake qualification reserved for refined-wake cases.',
        stop_method='Authenticated worker stop file; preserve accepted checkpoint and all frames.'))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['create','run','stop']);a.add_argument('name');v=a.parse_args();globals()[v.action](v.name)
