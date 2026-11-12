"""Record and package the bounded preparation optimization, retaining alpha134."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.build_unified_qgis import record,package,merge_bindings,verify,sha,write

REVIEWED={
    'braids_core/river_project.py','braids_core/sediment/adaptive_river_gis.py',
    'braids_qgis/core/raster_surface.py','braids_core/sediment/river_runtime_cache.py',
    'braids_core/sediment/river_runtime_lease.py','braids_gpu/general_live_transport.py',
}
ADDED={'braids_core/boundaries/raster_quadrature.py','braids_core/boundaries/raster_quadrature.h'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'dist/unified-alpha135')
    a=p.parse_args();base=ROOT/'validation/_preparation135';evidence=base/'release_evidence'
    evidence.mkdir(parents=True,exist_ok=False)
    old=ROOT/'dist/unified-alpha134/braids_qgis';manifest=verify(old)
    runtime=json.loads((old/'river_runtime/RUNTIME.json').read_text());changes=[]
    for relative in runtime['files']:
        before,after=old/'river_runtime'/relative,ROOT/relative
        if before.suffix not in ('.py','.h','.cu','.cpp','.f90','.f','.inc') or not after.is_file():continue
        if sha(before)!=sha(after):
            if relative not in REVIEWED:raise ValueError('Unreviewed runtime change: '+relative)
            changes.append(dict(path=relative,before=sha(before),after=sha(after)))
    for relative in sorted(ADDED):
        if relative in runtime['files']:raise ValueError('Expected new preparation implementation')
        changes.append(dict(path=relative,before=None,after=sha(ROOT/relative)))
    retained=ROOT/'validation/_performance134/release_evidence'
    prior=json.loads((retained/'BINDINGS.json').read_text())
    projects=ROOT/'validation/_performance134/release_projects'
    paths=[projects/f'check_{mode}_cpu_{physics}.json' for mode in ('dynamic','static') for physics in ('hydro','bedload')]
    # Exercise the added cropped-cell native preparation binding via the real
    # public route. Keep the matched geometry and all physical settings.
    source=ROOT/'validation/_performance134/current_inputs/cpu8_clock1_output1.json'
    doc=json.loads(source.read_text());doc['settings'].update(end_time_s=.1,dt_max_s=.1,output_interval_s=.1)
    extra=base/'record_cropped_cpu.json';write(extra,doc)
    write(base/'RECORD_INPUT.json',dict(source=str(source),source_sha256=sha(source),created=str(extra),
        created_sha256=sha(extra),changes={'settings.end_time_s':.1,'settings.dt_max_s':.1,'settings.output_interval_s':.1}))
    record(evidence,paths+[extra]);fresh=json.loads((evidence/'BINDINGS.json').read_text())
    write(evidence/'BINDINGS.json',merge_bindings(prior,fresh))
    rec=json.loads((evidence/'RECORD.json').read_text());oldrec=json.loads((retained/'RECORD.json').read_text())
    rec['seeds']=sorted(set(rec['seeds']+oldrec['seeds']))
    rec['retained_bindings']=dict(source=str(retained/'BINDINGS.json'),sha256=sha(retained/'BINDINGS.json'),
        previous_release_manifest_sha256=sha(old/'MANIFEST.json'),previous_source_identity=manifest['source_identity'],
        unchanged_existing_native_sources=True,reviewed_runtime_changes=changes,
        added_native_scope='Preparation-only exact bilinear polygon integration; no hydraulic/sediment kernel change',
        current_public_recordings=len(fresh),retained_unexercised=sorted(set(prior)-set(fresh)))
    write(evidence/'RECORD.json',rec);package(evidence,a.output.resolve())

if __name__=='__main__':main()
