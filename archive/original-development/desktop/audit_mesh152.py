"""Read-only reproduction and clean/profiled geometry timings for 01obstacle."""
from pathlib import Path
import argparse,cProfile,hashlib,io,json,pstats,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--runtime',required=True);parser.add_argument('--name',required=True)
parser.add_argument('--profile',action='store_true');args=parser.parse_args()
runtime=ROOT/args.runtime;sys.path[:0]=[str(runtime),str(ROOT)]
from braids_core.river_project import worker_profile,load_project
from braids_core.sediment.adaptive_river_worker import hierarchy
from braids_core.sediment.river_polygon_banks import topology
out=ROOT/'validation/_mesh152';out.mkdir(exist_ok=True)
source=Path('C:/Users/STREEM/Desktop/01obstacle/River_data/braids_project.json')
content=source.read_bytes();doc=load_project(source);profile=worker_profile(doc)
pr=cProfile.Profile();start=time.perf_counter()
if args.profile:pr.enable()
tree=hierarchy(profile)
if args.profile:pr.disable()
elapsed=time.perf_counter()-start
report=dict(source=str(source),source_sha256=hashlib.sha256(content).hexdigest(),runtime=str(runtime),seconds=elapsed,levels=[])
for mesh in tree.meshes:
    data=dict(area=np.array([c.area_m2 for c in mesh.cells]),owner=np.array([f.owner for f in mesh.faces]),
        neighbour=np.array([f.neighbour for f in mesh.faces]),normal=np.array([f.outward_normal for f in mesh.faces]).T,
        length=np.array([f.length_m for f in mesh.faces]))
    pairs={}
    for f in mesh.faces:
        if f.neighbour<0:continue
        key=tuple(sorted((f.owner,f.neighbour)));axis=int(np.argmax(np.abs(f.outward_normal)))
        pairs.setdefault(key,set()).add(axis)
    mixed=[pair for pair,axes in pairs.items() if len(axes)>1]
    row=dict(cells=len(mesh.cells),minimum_area=min(data['area']),mixed_axis_pairs=len(mixed),example=mixed[:5])
    try:topology(data,[c.vertices_xy for c in mesh.cells],transfers=False);row['topology']='pass'
    except ValueError as exc:row['topology']=str(exc)
    report['levels'].append(row)
np.savez(out/(args.name+'_finest.npz'),area=np.array([c.area_m2 for c in tree.finest.cells]),
    vertices=np.concatenate([np.asarray(c.vertices_xy) for c in tree.finest.cells]),
    offsets=np.r_[0,np.cumsum([len(c.vertices_xy) for c in tree.finest.cells])])
assert content==source.read_bytes()
(out/(args.name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
if args.profile:
    pr.dump_stats(str(out/(args.name+'.prof')));stream=io.StringIO();pstats.Stats(pr,stream=stream).sort_stats('cumtime').print_stats(55)
    (out/(args.name+'.txt')).write_text(stream.getvalue());print(stream.getvalue())
print(json.dumps(report,indent=2))
