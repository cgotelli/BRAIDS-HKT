"""Short matched-input Desktop/FLOW groin; validation decks only, no engine edits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT/'desktop/runtime'
sys.path[:0] = [str(RUNTIME), str(ROOT), str(ROOT/'validation/sediment/delft3d_current')]
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np

BASE = ROOT/'validation/_bedload161_matched'
PARENT = ROOT/'validation/sediment/river_morphodynamics/reports/widening_gap_v1/delft_dt0.01'
END, SPIN, FEED, STOCK = 120., 40., .0001, .1
SPECS = {
    'mpm': (2, dict(ACal=1.)),
    'ashida_michiue_generalized': (14, dict(ACal=17., ThetaC=.05, PowerM=1.5, PowerP=1., PowerQ=1.)),
    'wang_fredsoe': (13, dict(ACal=1., VicMol=1e-6)),
}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2)+'\n', encoding='utf-8')
def replace(text, key, value):
    pattern = r'(?mi)^\s*'+re.escape(key)+r'\s*=.*$'
    # FLOW's legacy MDF reader expects a six-column keyword field.
    line = f'{key:<6} = {value}'
    if re.search(pattern, text): return re.sub(pattern, line, text)
    return text+'\n'+line+'\n'

def table(name, parameter, unit, value, hydraulic=False):
    entries = [('table-name', f"'Groin {name}'"),('contents',"'Uniform             '"),
        ('location',f"'{name:<20}'"),('time-function',"'non-equidistant'"),
        ('reference-time','20000101'),('time-unit',"'minutes'"),('interpolation',"'linear'"),
        ('parameter',"'time                '                     unit '[min]'")]
    text = ''.join(f'{k:<21}{v}\n' for k,v in entries)
    for suffix in ('  end A', '  end B') if hydraulic else ('',):
        text += f"{'parameter':<21}'{parameter}{suffix}' unit '[{unit}]'\n"
    values = f'{value:.17g} '+(f'{value:.17g}' if hydraulic else '')
    return text+f"{'records-in-table':<21}2\n0 {values}\n{END/60:g} {values}\n"

def delft(formula, dt=.01):
    import runtime
    out = BASE/('delft_'+formula+'_v8'+('' if dt == .01 else f'_dt{dt:g}'))
    out.mkdir(parents=True, exist_ok=False)
    original = json.loads((PARENT/'INPUTS.json').read_text())
    for name, checksum in original['inputs'].items():
        assert sha(PARENT/name) == checksum, name
    shutil.copy2(PARENT/'config_flow2d3d.xml', out/'config_flow2d3d.xml')
    mdf = (PARENT/'sim_P057.mdf').read_text()
    for key, value in dict(MNKmax='42 18 1', Tstop=str(END), Dt=str(dt), Sub1='#   #',
        FilDry='#groin.dry#', Betac='0', Ccofu='45', Ccofv='45', Vicouv='0', Dicouv='0',
        Flmap=f'0 5 {END:g}', Flhis=f'0 5 {END:g}', Scour='#scour.dat#').items():
        mdf = replace(mdf, key, value)
    (out/'sim_P057.mdf').write_text(mdf)
    (out/'scour.dat').write_text('0\n1\n')
    (out/'groin.dry').write_text(''.join(f'18 {n} 21 {n}\n' for n in range(2,8)))
    (out/'flume.enc').write_text('1 1\n42 1\n42 18\n1 18\n1 1\n')
    xx, yy = np.meshgrid(np.arange(41)*.5, np.arange(17)*.5)
    grid = 'Coordinate System = Cartesian\n41 17\n0 0 0\n'
    for array in (xx, yy):
        for j, row in enumerate(array):
            for start in range(0, len(row), 5):
                grid += (f' ETA= {j+1:4d} ' if start == 0 else '           ')
                grid += ' '.join(f'{v:.17e}' for v in row[start:start+5])+'\n'
    (out/'flume.grd').write_text(grid)
    # DP bed values and initial fields are at water-level cell centres.
    xc = (np.arange(42)-.5)*.5
    bed = np.broadcast_to(.1-.001*xc, (18,42)).copy()
    np.savetxt(out/'bed.dep', -bed, fmt='%.17e')
    np.savetxt(out/'initial.ini', np.concatenate([np.full_like(bed,.6),np.zeros_like(bed),np.zeros_like(bed)]), fmt='%.17e')
    (out/'flume.bnd').write_text('upstream             T T     1     2     1    17 0.000000 Uniform\ndownstream           Z T    42     2    42    17 0.000000 Uniform\n')
    (out/'sections.crs').write_text('west                    1      2      1     17\neast                   41      2     41     17\n')
    (out/'flume.bct').write_text(table('upstream','total discharge (t)','m3/s',4.,True)+table('downstream','water elevation (z)','m',.58,True))
    (out/'bcm.bcm').write_text(table('upstream','transport excl pores Sediment1','m2/s',FEED/8)+table('downstream','transport excl pores Sediment1','m2/s',0.))
    # This locked FLOW release accepts v2; explicit near-degenerate bounds
    # avoid its implicit D90=1.5*D50 convention for an unspecified distribution.
    (out/'sed.sed').write_text('[SedimentFileInformation]\nFileVersion=02.00\n[SedimentOverall]\nCref=1600\nIopSus=0\n[Sediment]\nName=#Sediment1#\nSedTyp=bedload\nRhoSol=2650\nSedMinDia=.00099999999\nSedMaxDia=.00100000001\nCDryB=1590\nIniSedThick='+str(STOCK)+'\nFacDSS=1\n')
    (out/'mor.mor').write_text(f'''[MorphologyFileInformation]
FileVersion=02.00
[Morphology]
MorFac=1
MorStt={SPIN}
BedUpd=true
CmpUpd=true
UpdInf=true
EqmBc=true
DensIn=false
Thresh=.001
SedThr=.001
ThetSD=0
HMaxTH=0
AlfaBs=0
AlfaBn=0
ISlope=1
IHidExp=1
Sus=0
Bed=1
SusW=0
BedW=0
BcFil=bcm.bcm
[Underlayer]
IUnderLyr=1
[Numerics]
UpwindBedload=true
[Boundary]
Name=upstream
IBedCond=5
[Boundary]
Name=downstream
IBedCond=5
[Output]
Default=true
TranspType=2
BedTranspDueToCurrentsAtZeta=true
BedTranspDueToCurrentsAtFlux=true
RawTransportsAtZeta=true
Taub=true
Taurat=true
''')
    law, parameters = SPECS[formula]
    (out/'tra.tra').write_text('[TransportFormulaFileInformation]\nFileVersion=01.00\n[TransportFormula]\nNumber='+str(law)+'\n'+''.join(f'{k}={v:.17g}\n' for k,v in parameters.items()))
    save(out/'INPUTS.json',dict(parent=str(PARENT), parent_manifest_sha256=sha(PARENT/'INPUTS.json'),
        changes=dict(geometry='20 x 8 m; 2 x 3 m bank-attached groin at x8..10; 0.5 m cells',
                     Q=4.,stage0=.6,outlet_stage=.58,chezy=45.,diameter=.001,d90=.001,stock=STOCK,
                     spin_s=SPIN,end_s=END,feed_solid_m3_s=FEED,feed_native_solid_m2_s=FEED/8,
                     no_secondary=True,no_viscosity=True,no_suspension=True,no_slope=True,no_bank_erosion=True,
                     outlet='prescribed water elevation with hydraulic backflow; zero sediment supply only on local inward water; free bedload export otherwise',
                     dt_s=dt,capacity=dict(method=formula,parameters=parameters)),
        Delft3D_source_modified=False,Delft3D_runtime_modified=False,
        inputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    result = runtime.run_case(out,configuration='config_flow2d3d.xml',timeout_s=180)
    print(json.dumps(result),flush=True)

def braids(formula, backend, dt=.1):
    from PySide6.QtWidgets import QApplication
    from check_boundaries159 import document,services
    from braids_desktop.forms import SettingsPanel
    from braids_desktop.river_physics import set_choice
    from braids_core.river_boundary import TOTAL
    from braids_core.river_open_boundary import resolve
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    from braids_core.river_worker import run
    from braids_desktop.preparation import prepare
    verify(); install(json.loads((RUNTIME/'PRECOMPILED.json').read_text()),RUNTIME)
    app=QApplication.instance() or QApplication([])
    out=BASE/('braids_'+formula+'_'+backend+('' if dt==.1 else f'_dt{dt:g}'))
    out.mkdir(parents=True,exist_ok=False)
    doc=document(out/'project',backend,'stage')
    f=SettingsPanel();f.load_document(doc)
    f.sediment_enabled.setChecked(True);f.physics.bedload_on.setChecked(True)
    f.physics.grains.item(0,1).setText('1')
    f.physics.layers.item(0,0).setText(str(STOCK))
    set_choice(f.physics.capacity,'capacity.'+formula)
    f.end_time.setValue(END);f.sediment_start.setValue(SPIN);f.morfac.setValue(1.)
    f.dt_max.setValue(dt)
    f.output_interval.setValue(5.)
    set_choice(f.forcing.method,TOTAL);f.forcing.table.item(0,2).setText(str(FEED))
    doc=f.document();f.close()
    # Explicit laboratory initial condition, not the public dry-start default.
    doc['initial'].update(stage_m=.6,velocity_m_s=[0.,0.])
    doc['boundary_states']['Downstream']['stage_policy']='stage.subcritical_backflow'
    doc.pop('editor_state',None)
    doc['boundary_resolution']=resolve(doc['settings']['hydraulics'],doc['boundary_states'],
        dict(Upstream='inlet',Downstream='outlet'),sediment=doc['model']['physics']['sediment'],ng=1,end_time=END)
    services.save_project_folder(out/'project',doc)
    prepare(out/'project/braids_project.json',out/'prepared',asset_cache=BASE/'assets')
    status=run(out/'prepared',out/'run',until=END)
    save(out/'STATUS.json',dict(status=status,runtime_manifest_sha256=sha(RUNTIME/'RUNTIME.json')))
    print(json.dumps(status),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('engine',choices=['braids','delft']);p.add_argument('formula',choices=SPECS)
    p.add_argument('--backend',choices=['cpu','gpu'],default='cpu');p.add_argument('--dt',type=float)
    a=p.parse_args()
    if a.engine=='delft':delft(a.formula,.01 if a.dt is None else a.dt)
    else:braids(a.formula,a.backend,.1 if a.dt is None else a.dt)
