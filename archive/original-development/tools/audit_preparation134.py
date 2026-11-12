"""Read-only preparation/clock audit of completed performance134 evidence.

This does not import or execute the solver and does not modify its sources.
"""
import json
from pathlib import Path
import re
import pstats
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation/_performance134'


def stats(a):
    return dict(count=len(a),minimum=min(a),maximum=max(a),mean=statistics.mean(a),median=statistics.median(a))


def braids(name):
    root=OUT/name
    p=json.loads((root/'PROFILE.json').read_text())
    index=json.loads((root/'run/frames/INDEX.json').read_text())
    rows=p['steps'];d=[r['diagnostics'] for r in rows]
    count=sum(r['hydraulic_substeps'] for r in d)
    clock=sum(r['outer_dt_s'] for r in rows)
    return dict(profile=str(root/'PROFILE.json'),backend=p['backend'],runtime_id=p['runtime_identity'],
                mesh='fixed level-one 3.25 m Cartesian/cropped mesh; 5624 cells',cfl=p['controls']['cfl'],
                settings=p['settings'],frames=len(index['frames']),frame_times_s=[r['time_s'] for r in index['frames']],
                outer_transactions=len(rows),outer_interval_s=stats([r['outer_dt_s'] for r in rows]),
                accepted_hydraulic_substeps=count,accepted_hydraulic_mean_dt_s=clock/count,
                accepted_hydraulic_minimum_dt_s=min(r['hydraulic_minimum_dt'] for r in d),
                accepted_hydraulic_maximum_dt_s=max(r['hydraulic_maximum_dt'] for r in d),
                last_next_substep_hint_s=d[-1]['hydraulic_next_substep_hint_s'],
                endpoint_remainders=sum(r['hydraulic_endpoint_remainders'] for r in d),
                internal_retries=sum(len(r['hydraulic_rejections']) for r in d),
                accepted_cfl_max=max(r['accepted_cfl_max'] for r in d),
                fwave_evaluations=sum(r['fwave_evaluations'] for r in d),
                first_interval=rows[0],last_interval=rows[-1],
                note='Count, total time and mean are exact from accepted intervals. Only per-interval min/max, not the complete individual hydraulic-step list, were recorded; no hydraulic-step median is inferred.')


def main():
    evidence={}
    for name in ('profile_cpu','profile_gpu'):
        if (OUT/name/'run/STATUS.json').exists():evidence[name]=braids(name)
    src=Path('C:/Users/STREEM/Desktop/DEM-project-CIVIL410/test.gid/proceso.rep')
    rows=[]
    for line in src.read_text().splitlines():
        m=re.match(r'\s*([\d.]+)\s+([\d.]+)\s+(\d+):(\d+):(\d+):(\d+)\s+([\d.]+)\s+([\d.]+)',line)
        if m:
            t,dt,h,mi,s,cs,qi,qo=map(float,m.groups())
            rows.append(dict(time_s=t,reported_dt_s=dt,wall_clock_s=3600*h+60*mi+s+.01*cs,qin_m3_s=qi,qout_m3_s=qo))
            if t>=60:break
    elapsed=rows[-1]['wall_clock_s']-rows[0]['wall_clock_s']
    evidence['iber']=dict(source=str(src),mesh='5189 triangular cells; median side4.9802m; median area10.68724m2',
        cfl=.45,requested_threads=1,output_interval_s=1.,dt_max_s=1.,sampled_rows=rows,
        sampled_dt=stats([r['reported_dt_s'] for r in rows[1:]]),last_time_s=rows[-1]['time_s'],
        reported_elapsed_s=elapsed,note='Approximately one output/log row per simulated second, not all accepted timesteps. The last dt can be an event remainder. Mean/median of these samples are not exact accepted-step averages.')
    profile=pstats.Stats(str(OUT/'profile_cpu/PREPARATION.pstats'))
    wanted={'prepare','prepare_recipe','surface_averages','cell_average','polygon_moments','polygon',
            'Intersection','prepare_shared','prepare_sources','_verify_sources','source_identity','hierarchy',
            'write_json_atomic','_io.open'}
    functions=[]
    for (path,line,name),(primitive,calls,self_time,cumulative,callers) in profile.stats.items():
        if name in wanted or '_io.open' in name:
            functions.append(dict(file=path,line=line,name=name,calls=calls,self_s=self_time,cumulative_s=cumulative))
    functions.sort(key=lambda row:row['cumulative_s'],reverse=True)
    evidence['preparation']=dict(instrumented_only=True,profile_total_s=profile.total_tt,functions=functions,
        warning='Cumulative timings nest. Do not add cell_average to surface_averages or _verify_sources to prepare_sources. Clean end-to-end timings govern speed claims.',
        priorities=[
            dict(rank=1,target='Exact cropped-cell DEM quadrature',measured='surface_averages7.999s including cell_average7.587s for481cells;60777moment evaluations',
                proposal='Retain exact bilinear integration; batch or native-compile polygon clipping and moment accumulation, reuse pixel-cell geometry/weights. Process independent cells in a measured parallel loop.',
                dependencies='DEM geotransform, cell vertices/holes, clipped-pixel support and valid-node masks determine reusable quadrature weights; elevations/scale/offset are live values, not cache identity substitutes.',
                gates='Constant/linear/bilinear analytic surfaces; rotated concave/holed and tiny wall crops; pixel-edge/outer-half-pixel geometry; NoData/CRS rejection; exact covered area; unchanged inlet terrain semantics; cancellation. Compare prepared elevations and short coupled results before speed adoption.'),
            dict(rank=2,target='Source archive construction and repeated validation',measured='prepare_sources5.095s includes _verify_sources4.094s;2378SHAcalls;9095file opens4.234s globally',
                proposal='Extend authenticated immutable source leasing into preparation, share already verified source bytes/digests across copy and check, and reuse a release-scoped content-addressed archive. Keep independent target integrity and inventory verification.',
                dependencies='Installed release identity, source inventory, generated-bridge transformation and native library contract; private destination publication must finish before reuse.',
                gates='Concurrent source/target mutation, same-size/mtime changes, added/deleted files, path replacement/reparse points, interrupted copying, cancellation, release changes, restart and moved-folder tests. Do not replace hashes with stat-only trust or remove required source provenance.'),
            dict(rank=3,target='Repeated hierarchy construction within one preparation',measured='hierarchy0.349s cumulative in2calls',
                proposal='Pass the already validated hierarchy from GIS sampling to core preparation rather than rebuilding identical geometry. Cross-process run reconstruction is a separate lower-priority cache question.',
                dependencies='Exact domain, open assignments, lattice, levels and no intervening mutation; immutable or detached hierarchy ownership.',
                gates='Identical cells/faces/IDs/areas/refinement requirements/native records, no mutable sharing, cancellation and corrupt identity rejection.'),
            dict(rank=4,target='Preparation progress publication',measured='230atomic JSON writes0.291s;107terrain progress callbacks',
                proposal='Coalesce duplicated progress reports at a bounded wall-clock cadence while publishing phase changes and completion immediately. Keep cancellation polling independent and frequent.',
                dependencies='Progress is UI state, not checkpoint durability or physical scheduling.',
                gates='Monotone percentage, every phase visible, terminal states durable, Stop still responsive; no skipped checkpoint/source validation.'),
        ])
    target=OUT/'PREPARATION_CLOCK_AUDIT.json'
    target.write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:{n:v for n,v in row.items() if n not in ('sampled_rows','frame_times_s','first_interval','last_interval')} for k,row in evidence.items() if k!='preparation'},indent=2))


if __name__=='__main__':main()
