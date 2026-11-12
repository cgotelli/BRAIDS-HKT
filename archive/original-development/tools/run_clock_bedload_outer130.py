"""Bounded, genuine outer-transaction sensitivity for one-grain bedload.

The fixed mesh has no AMR assessment clock. Outputs at 2.5 and 5 seconds do
not conceal the requested 2-second morphology/outer transaction. Hydraulic
SSPRK22 stages retain face-sum CFL0.5, independent of the outer transaction.
No maintained runtime is edited and no long-term morphology claim follows.
"""
from contextlib import contextmanager
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"tests"))
from test_river_exterior_inlet_integration import document
from braids_core import river_worker
from braids_core.river_open_boundary import resolve
from braids_core.sediment.shoreline_solver import ShorelineSolver
from braids_core.sediment.bed_columns_v101 import host
from braids_gpu.atomic_output import write_json_atomic
from validation.hydraulics.speed130_clock import candidate


def terrain(path):
    from osgeo import gdal,osr
    gdal.UseExceptions()
    ds=gdal.GetDriverByName("GTiff").Create(str(path),60,40,1,gdal.GDT_Float64)
    ds.SetGeoTransform((-1,.1,0,3,0,-.1))
    srs=osr.SpatialReference();srs.ImportFromEPSG(2056);ds.SetProjection(srs.ExportToWkt())
    x=-1+(np.arange(60)+.5)*.1
    ds.GetRasterBand(1).WriteArray(np.broadcast_to(-.001*x,(40,60)).copy())
    ds.FlushCache();ds=None


def project(dem,backend,outer):
    d=document(dem,backend,"stage_coupled");p=d["settings"]
    p.update(adaptive=False,levels=1,end_time_s=5.,dt_max_s=outer,output_interval_s=2.5)
    p["controls"]["cfl"]=.5;p["adaptation"]["terrain_enabled"]=True
    d["name"]=f"Clock130 bedload outer{outer:g} {backend}"
    d["boundary_resolution"]=resolve(p["hydraulics"],d["boundary_states"],
        dict(inlet="inlet",outlet="outlet"),sediment=d["model"]["physics"]["sediment"],ng=1,end_time=5.)
    return d


def load_frames(folder):
    index=json.loads((folder/"frames/INDEX.json").read_text());frames=[]
    for row in index["frames"]:
        with np.load(folder/"frames"/row["file"],allow_pickle=False) as f:
            state={k:f[k].copy() for k in f.files}
        with np.load(folder/"frames"/row["epoch"],allow_pickle=False) as f:
            mesh={k:f[k].copy() for k in f.files}
        frames.append((float(row["time_s"]),state,mesh))
    return frames


def compare(a,b):
    rows=[]
    for (ta,qa,ma),(tb,qb,mb) in zip(a,b):
        assert ta==tb
        assert set(ma)==set(mb) and all(np.array_equal(ma[k],mb[k]) for k in ma)
        area=ma["area"];dh=qa["q"][0]-qb["q"][0];dz=qa["z"]-qb["z"]
        rows.append(dict(time_s=ta,depth_max_m=float(abs(dh).max()),
            depth_rms_m=float(np.sqrt(np.average(dh*dh,weights=area))),
            bed_max_m=float(abs(dz).max()),bed_rms_m=float(np.sqrt(np.average(dz*dz,weights=area)))))
    assert len(a)==len(b)==len(rows)
    return dict(per_time=rows,**{k:max(r[k] for r in rows) for k in ("depth_max_m","depth_rms_m","bed_max_m","bed_rms_m")})


def run(dem,backend,outer,folder):
    folder.mkdir();write_json_atomic(folder/"PROJECT.json",project(dem,backend,outer))
    prepared=river_worker.prepare(folder/"PROJECT.json",folder/"prepared")
    steps=[];original=ShorelineSolver.step
    def step(self,dt,*args,**kwargs):
        result=original(self,dt,*args,**kwargs)
        steps.append(dict(time_s=self.time,outer_dt_s=dt,diagnostics=result["diagnostics"],
            minimum_depth_m=float(host(self.q)[0].min()),max_depth_m=float(host(self.q)[0].max())))
        return result
    try:
        ShorelineSolver.step=step
        with candidate("combined") as clock:
            status=river_worker.run(prepared,folder/"run")
    finally:ShorelineSolver.step=original
    frames=load_frames(folder/"run")
    rows=json.loads((folder/"run/DIAGNOSTICS.json").read_text())
    water=max(abs(r["water_residual_m3"]) for r in rows)
    solid=max(abs(v) for r in rows for v in r["class_residual_kg"])
    wall=max(abs(v) for r in rows for v in r["wall_integrals"])
    checks=dict(complete=status["status"]=="complete",water_conservative=water<1e-10,
        sediment_conservative=solid<1e-8,walls_closed=wall==0.,
        nonnegative_depth=all(r["minimum_depth_m"]>=0 for r in steps),
        hard_stage_cfl=status["hydraulic"]["accepted_cfl_max"]<=.5*(1+1e-12),
        actual_requested_outer_reached=max(r["outer_dt_s"] for r in steps)==outer,
        no_silent_outer_rejection=not status["rejected_trials"])
    result=dict(status=status,clock=clock,outer_dt_requested_s=outer,backend=backend,
        steps=steps,checks=checks,max_water_residual_m3=water,max_sediment_residual_kg=solid,
        max_actual_bed_change_m=max(float(abs(f[1]["z"]-frames[0][1]["z"]).max()) for f in frames),
        scope="5 s fixed-mesh one-grain bedload; actual large physical outer updates, not long morphology validation",
        timings_qualified=False)
    write_json_atomic(folder/"EVIDENCE.json",result)
    return result,frames


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--analyze-only",action="store_true");a=p.parse_args();a.output=a.output.resolve()
    if not a.analyze_only:
        a.output.mkdir(parents=True,exist_ok=False);terrain(a.output/"dem.tif")
    baseline=river_worker.source_identity();results={};frames={}
    for label,outer in (("outer2",2.),("outer05",.5),("outer0125",.125)):
        for backend in ("cpu","gpu"):
            name=backend+"_"+label;folder=a.output/name
            if a.analyze_only:
                results[name]=json.loads((folder/"EVIDENCE.json").read_text());frames[name]=load_frames(folder/"run")
            else:
                print("START",name,flush=True)
                results[name],frames[name]=run(a.output/"dem.tif",backend,outer,folder)
                print("DONE",name,json.dumps(results[name]["checks"]),flush=True)
    pairs={}
    for backend in ("cpu","gpu"):
        for label in ("outer2","outer05"):
            pairs[backend+"_"+label+"__fine"]=compare(frames[backend+"_"+label],frames[backend+"_outer0125"])
    for label in ("outer2","outer05","outer0125"):
        pairs[label+"__cpu_gpu"]=compare(frames["cpu_"+label],frames["gpu_"+label])
    write_json_atomic(a.output/"SUMMARY.json",dict(runtime=baseline,runtime_after=river_worker.source_identity(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),comparisons=pairs,
        cases={name:{k:v for k,v in r.items() if k!="steps"} for name,r in results.items()},
        production_changed=False,timings_qualified=False))
    assert baseline==river_worker.source_identity()
    print(json.dumps(pairs,indent=2))


if __name__=="__main__":main()
