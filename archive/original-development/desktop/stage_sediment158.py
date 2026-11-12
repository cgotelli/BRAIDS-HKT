"""Isolated candidate and native packages; do not alter installed runtime."""
import hashlib,json,shutil,sys,inspect
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_sediment158';TARGET=OUT/'candidate'
FILES=('braids_core/river_material_zones.py','braids_core/river_project.py','braids_core/river_worker.py',
 'braids_core/river_boundary.py','braids_core/river_boundary.h',
 'braids_core/sediment/adaptive_river_worker.py','braids_core/sediment/river_roughness.py',
 'braids_core/sediment/river_hydraulic_drag.py','braids_core/sediment/river_exterior_sediment.py',
 'braids_core/sediment/river_exterior_bed.py','braids_core/sediment/river_explicit.py',
 'braids_core/sediment/bed_columns_v101.py','braids_core/sediment/class_bed_update.py',
 'braids_core/sediment/class_bed_update.h')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    if not TARGET.exists():shutil.copytree(ROOT/'desktop/runtime',TARGET)
    manifest=json.loads((TARGET/'RUNTIME.json').read_text())
    for name in FILES:
        shutil.copy2(ROOT/name,TARGET/name);manifest['files'][name]=sha(TARGET/name)
    (TARGET/'RUNTIME.json').write_text(json.dumps(manifest,indent=2)+'\n')
    sys.path.insert(0,str(TARGET))
    from validation.sediment.river_morphodynamics import sediment_wave
    from braids_core.sediment.portable_river_runtime import _bindings,_key
    from braids_core.sediment import river_roughness,river_hydraulic_drag
    generated=OUT/'generated';generated.mkdir(parents=True,exist_ok=True)
    r=generated/'spatial_roughness.h';r.write_text(river_roughness.source(True))
    d=generated/'spatial_drag.h';d.write_text(river_hydraulic_drag.source(True))
    table=json.loads((TARGET/'PRECOMPILED.json').read_text())
    from braids_core.sediment.class_bed_update import NAMES as class_names
    for source,names in ((TARGET/'braids_core/river_boundary.h',('river_prescribed_boundary',)),(r,river_roughness.NAMES),(d,('river_cell_drag',)),(TARGET/'braids_core/sediment/class_bed_update.h',class_names)):
        engine=sediment_wave.WaveEngine('cpu',2,source=source,names=names)
        bound=inspect.signature(sediment_wave.WaveEngine.__init__).bind(engine,'cpu',2,source=source,names=names);bound.apply_defaults()
        key=_key(sediment_wave,'WaveEngine',bound.arguments);row=_bindings(engine)
        for name,file in row['libraries'].items():
            src=Path(file);folder=TARGET/'precompiled'/sha(src)[:20];folder.mkdir(parents=True,exist_ok=True)
            for path in (src,*[Path(shutil.which('g++')).parent/n for n in ('libgomp-1.dll','libwinpthread-1.dll','libgcc_s_seh-1.dll','libdl.dll')]):
                target=folder/path.name
                if target.exists():assert sha(target)==sha(path)
                else:shutil.copy2(path,target)
                manifest['files'][target.relative_to(TARGET).as_posix()]=sha(target)
            row['libraries'][name]=dict(file=(folder/src.name).relative_to(TARGET).as_posix(),sha256=sha(src))
        table[key]=row;print(key,flush=True)
    (TARGET/'PRECOMPILED.json').write_text(json.dumps(table,indent=2)+'\n');manifest['files']['PRECOMPILED.json']=sha(TARGET/'PRECOMPILED.json')
    (TARGET/'RUNTIME.json').write_text(json.dumps(manifest,indent=2)+'\n');print(sha(TARGET/'RUNTIME.json'))
if __name__=='__main__':main()
