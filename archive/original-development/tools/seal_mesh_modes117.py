"""Finalize release evidence without changing any simulation or installed file."""
import hashlib,json,sys,zipfile,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from braids_core.river_worker import source_identity
BASE=ROOT/'validation/_mesh_modes117'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/unified_qgis_v1/mesh_modes117'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    checks=read(BASE/'portable_staged118/CHECKS.json');pairs={};budget={}
    for mode,i in [('dynamic',0),('static',2),('dynamic_bedload',4),('static_bedload',6)]:
        names=list(checks['runs']);cpu=Path(checks['runs'][names[i]]);gpu=Path(checks['runs'][names[i+1]])
        ic=read(cpu/'frames/INDEX.json');ig=read(gpu/'frames/INDEX.json');result={k:0. for k in ('q','z','mass','W')}
        for a,b in zip(ic['frames'],ig['frames']):
            assert abs(a['time_s']-b['time_s'])<1e-12
            with np.load(cpu/'frames'/a['epoch']) as x,np.load(gpu/'frames'/b['epoch']) as y:
                np.testing.assert_array_equal(x['leaves'],y['leaves'])
            with np.load(cpu/'frames'/a['file']) as x,np.load(gpu/'frames'/b['file']) as y:
                for k in result:result[k]=max(result[k],float(np.max(abs(x[k]-y[k]))))
        assert result['q']<1e-8 and result['z']<1e-8 and result['mass']<1e-7
        pairs[mode]=result
        for path in (cpu,gpu):
            rows=read(path/'DIAGNOSTICS.json');events=read(path/'ADAPTATION.json')
            budget[path.name]=dict(water_residual_m3=max(abs(r['water_residual_m3']) for r in rows),
                class_residual_kg=max(max(abs(v) for v in r['class_residual_kg']) for r in rows),
                wall_integrals=max(max(abs(v) for v in r['wall_integrals']) for r in rows),
                assessments=len(events),regrids=sum(bool(e.get('changed')) for e in events))
    ui=read(BASE/'qgis_installed118/CHECKS.json');install=read(BASE/'INSTALL_ALPHA118.json')
    assert all(checks['checks'].values()) and all(ui['checks'].values()) and install['status']=='installed'
    record=read(BASE/'release_record/RECORD.json')
    delivery=dict(package=read(BASE/'release_record/PACKAGE.json'),installation=install,
        recorded_runs=len(record['projects']),native_bindings=len(read(BASE/'release_record/BINDINGS.json')),
        supplemental_cpu_recordings=len(read(BASE/'clock_record/RECORD.json')['projects']),
        portable_checks=len(checks['checks']),installed_qgis_checks=len(ui['checks']),
        cpu_gpu_maximum_absolute_differences=pairs,budgets=budget)
    (OUT/'DELIVERY.json').write_text(json.dumps(delivery,indent=2))
    for name in ('MESH_DYNAMIC.png','MESH_STATIC.png'):shutil.copy2(BASE/'qgis_installed118'/name,OUT/name)
    sources=[
        'braids_core/river_project.py','braids_core/river_worker.py',
        'braids_core/sediment/adaptive_river_gis.py','braids_core/sediment/adaptive_river_worker.py',
        'braids_core/sediment/river_field_policy.py','braids_core/sediment/river_midpoint.py','braids_core/sediment/river_terrain_refinement.py',
        'braids_core/sediment/river_terrain_refinement.h','braids_core/sediment/river_kernel_cache.py',
        'validation/sediment/river_morphodynamics/sediment_wave.py',
        'braids_qgis/gui/river_dock.py','braids_qgis/metadata.txt','braids_qgis/FIRST_RELEASE.md','braids_qgis/MODEL_SETUP.md',
        'validation/hydraulics/smallcell117/redistribution.py','validation/hydraulics/smallcell117/redistribution.h',
        'tools/profile_smallcell117.py','tools/profile_mesh_modes117.py','tools/prepare_mesh_modes117.py',
        'tools/record_mesh_modes117.py','tools/analyze_mesh_modes117.py','tools/seal_mesh_modes117.py',
        'tools/build_unified_qgis.py','tools/qualify_clock117.py','tools/diagnose_mesh_modes117.py',
        'tests/test_river_mesh_modes.py','tests/test_river_terrain_refinement.py','tests/test_river_depth_policy.py',
        'tests/test_river_project.py','tests/test_river_public_boundaries.py','tests/test_river_public_survey.py',
        'tests/test_river_midpoint_retry_cap.py',
        'tests/support/qgis_mesh_modes117.py','README.md','validation/RIVER_MORPHODYNAMICS_STATUS.md',
        'validation/sediment/river_morphodynamics/UNIFIED_QGIS_PLAN.md']
    with zipfile.ZipFile(OUT/'SOURCES.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in sources:z.write(ROOT/name,name)
    artifacts=[p for p in OUT.iterdir() if p.is_file() and p.name!='MANIFEST.json']
    artifacts+=list(BASE.glob('*.xml'))+list(BASE.glob('*.json'))+[BASE/'BEFORE_SOURCES.zip']
    for folder in ('cpu_dynamic','gpu_dynamic','cpu_static','gpu_static','smallcell_cpu','smallcell_gpu'):
        path=BASE/folder;artifacts+=list(path.glob('*.json'))+list((path/'run/frames').glob('*'))
    artifacts += [BASE/'portable_staged118/CHECKS.json',BASE/'qgis_staged118/CHECKS.json',BASE/'qgis_installed118/CHECKS.json',
        BASE/'release_record/RECORD.json',BASE/'release_record/BINDINGS.json',BASE/'clock_record/RECORD.json',BASE/'clock_record/BINDINGS.json']
    (OUT/'MANIFEST.json').write_text(json.dumps(dict(source_identity=source_identity(),
        files={p.relative_to(ROOT).as_posix():sha(p) for p in artifacts if p.is_file()}),indent=2))
    print(json.dumps(delivery,indent=2))

if __name__=='__main__':main()
