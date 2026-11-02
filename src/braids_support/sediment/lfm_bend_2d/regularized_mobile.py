"""Guarded, finite continuation after hydraulic qualification; no progress polling.

Only the isolated BRAIDS candidate is executed. The completed stock Delft3D
reference is opened read-only for the final same-resolution comparison.
"""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import numpy as np
from . import run_case as r
from . import analyze_runs as a
from . import summarize_regularization as summary

HYDRO_TAG='regularized_cold300_v1'
RUNTIME_TAG='fine_regularized_v1'
DELFT_TAG='fine3600_m1_v1_mobile_delft'


def completed_hydraulics(report):
    status=json.loads((report/'status.json').read_text())
    if status.get('status')!='complete':raise RuntimeError(f'Hydraulic qualification not complete: {status}')
    record=json.loads((report/'qualification.json').read_text())
    cases=record.get('cases',[])
    if len(cases)!=2 or record.get('hydraulic_gates_passed') is not True:
        raise RuntimeError('Both hydraulic resolutions must pass before sediment transport')
    for case in cases:
        gates=case.get('gates',{})
        required=('steady_stage','inlet','outlet','positive_depth','water_budget',
                  'secondary_budget','momentum_budget','secondary_active')
        if any(gates.get(key) is not True for key in required):
            raise RuntimeError(f'Hydraulic gate failed or missing: {case.get("label")}')
    return record


def wait_for_supervisor(pid):
    """One Windows process-handle wait, not a loop reading run status/logs."""
    if os.name!='nt':raise RuntimeError('Process-handle continuation is Windows-only')
    if pid<=0 or pid==os.getpid():raise ValueError('Invalid supervisor PID')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.QueryFullProcessImageNameW.argtypes=[wintypes.HANDLE,wintypes.DWORD,wintypes.LPWSTR,ctypes.POINTER(wintypes.DWORD)]
    kernel.QueryFullProcessImageNameW.restype=wintypes.BOOL
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
    kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle.restype=wintypes.BOOL
    handle=kernel.OpenProcess(0x00100000|0x1000,False,pid)  # synchronize/query limited
    if not handle:
        # The supervisor may have exited between our status check and OpenProcess.
        if ctypes.get_last_error()==87:return
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        buffer=ctypes.create_unicode_buffer(32768);size=wintypes.DWORD(len(buffer))
        if not kernel.QueryFullProcessImageNameW(handle,0,buffer,ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        if Path(buffer.value).resolve()!=Path(sys.executable).resolve():
            raise RuntimeError('Supervisor PID no longer refers to the expected Python executable')
        result=kernel.WaitForSingleObject(handle,0xFFFFFFFF)
        if result!=0:raise RuntimeError(f'Process wait failed: {result}')
    finally:kernel.CloseHandle(handle)


def locked_candidate(runtime):
    identity=json.loads((runtime/'build.json').read_text())
    exe=runtime/'xbraids_candidate.exe'
    if r.sha256(exe)!=identity['executable_sha256']:raise ValueError('Candidate executable changed')
    if identity['depth_factor']!=2.:raise ValueError('Unexpected regularization depth factor')
    for name,digest in identity['source_sha256'].items():
        if (runtime/Path(name).name).is_file():
            source=runtime/Path(name).name
        else:source=r.ROOT/name
        if r.sha256(source)!=digest:raise ValueError(f'Candidate source receipt mismatch: {name}')
    return exe,identity


def verify_common_configuration(native,delft):
    keys=('dx','Q','manning','feed_kg_s','porosity','d50_m','d90_m',
          'viscosity_m2_s','secondary_momentum_beta','longitudinal_slope_factor',
          'transverse_slope_factor','morfac')
    for key in keys:
        if not np.isclose(native[key],delft[key],rtol=1e-12,atol=0):
            raise ValueError(f'Unmatched physical/run setting: {key}')
    if delft['nk']!=1:raise ValueError('Delft3D reference must be depth-averaged')


def mobile_gates(metrics,accounts):
    """Operational screen, not evidence of correct erosion/deposition patterns."""
    return dict(positive_depth=metrics['min_depth_m']>0,
        inlet_Q=metrics['inlet_discharge_relative_error']<.005,
        full_sediment_feed=abs(metrics['prescribed_feed_fraction']-1)<.005,
        water_budget=abs(metrics['water_budget_residual_m3'])<1e-6,
        sediment_budget=metrics['sediment_budget_residual_over_throughput']<1e-6,
        sediment_export=metrics['exported_sediment_kg']>0,
        secondary_budget=accounts['secondary_inventory_budget_max_residual']<1e-8,
        momentum_budget=accounts['momentum_budget_max_residual']<1e-8,
        # Predeclared gross-artifact guards, not a claim that smaller artefacts
        # cannot exist. A failed guard stops for scientific review, not tuning.
        no_gross_cell_scale_bed_noise=metrics['local_plane_residual_rms_m']<.002,
        no_gross_outlet_deposit=metrics['outlet_max_deposition_m']<.01)


def hydraulic_mesh_screen(coarse,fine):
    """Predeclared interior profile consistency, not a convergence proof."""
    from . import experiment as e
    n=np.tile(np.linspace(.1,.9,25)*e.WIDTH,3)
    s=np.repeat(e.APPROACH+e.RADIUS*np.deg2rad([45.,90.,135.]),25)
    rows={}
    for field,values,limit in [('speed',[coarse['speed'][-1],fine['speed'][-1]],.05),
                             ('secondary_intensity',[summary.intensity(coarse),summary.intensity(fine)],.10)]:
        left,out_left=a.sample(coarse,values[0],s,n)
        right,out_right=a.sample(fine,values[1],s,n)
        rms=float(np.sqrt(np.mean((left-right)**2)))
        scale=float(np.sqrt(np.mean(.5*(left**2+right**2))))
        relative=rms/max(scale,1e-12)
        rows[field]=dict(rms_difference_m_s=rms,relative_rms_difference=relative,
            limit=limit,extrapolated_points=int((out_left|out_right).sum()),
            passed=bool(relative<=limit and not (out_left|out_right).any()))
    return dict(fields=rows,passed=all(row['passed'] for row in rows.values()),
        criterion='Interior 45/90/135-degree profiles: speed <=5%, secondary intensity <=10% relative RMS',
        limitation='Operational coarse/fine consistency only; no experimental accuracy or grid convergence proof')


def artifact_figure(data,report):
    """Expose cell-scale bed variation and outlet profiles without smoothing dz."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from scipy.spatial import cKDTree
    from . import experiment as e
    residuals=[]
    for d in data:
        dz=d['z'][-1]-d['z'][0]
        _,neighbors=cKDTree(d['xy']).query(d['xy'],k=9)
        values=np.empty(len(dz))
        for i,ids in enumerate(neighbors):
            matrix=np.column_stack([np.ones(len(ids)),d['xy'][ids]-d['xy'][i]])
            values[i]=dz[i]-np.linalg.lstsq(matrix,dz[ids],rcond=None)[0][0]
        residuals.append(values*1000)
    limit=max(.1,max(float(np.max(abs(values))) for values in residuals))
    fig,axes=plt.subplots(2,len(data),figsize=(6*len(data),8),squeeze=False,layout='constrained')
    for j,(d,values) in enumerate(zip(data,residuals)):
        ax=axes[0,j]
        im=PolyCollection(d['polygons'],array=values,cmap='RdBu_r',clim=(-limit,limit),edgecolors='none')
        ax.add_collection(im);ax.autoscale();ax.set_aspect('equal')
        ax.set(title=f"{d['p']['engine'].upper()}: local-plane residual",xlabel='x (m)',ylabel='y (m)')
        fig.colorbar(im,ax=ax,label='Bed-change residual (mm)',shrink=.8)
        ax=axes[1,j];s=np.linspace(e.LENGTH-3,e.LENGTH-d['p']['dx'],80)
        for relative in (.2,.5,.8):
            values,_=a.sample(d,d['z'][-1]-d['z'][0],s,np.full_like(s,relative*e.WIDTH))
            ax.plot(e.LENGTH-s,values*1000,label=f'{relative:g} width from outer bank')
        ax.axhline(0,color='gray',lw=.7);ax.invert_xaxis();ax.grid(alpha=.2);ax.legend(fontsize=8)
        ax.set(title='Outlet approach:\nsampled raw bed-change profiles',
               xlabel='Distance upstream of outlet (m)',ylabel='Bed change (mm)')
    fig.suptitle('Cell-scale and outlet diagnostics\nPlane residual also contains physical curvature;\nit is not proof of checkerboarding',fontsize=11)
    fig.savefig(report/'bed_scale_outlet.png',dpi=160);plt.close(fig)


def run(tag,after_pid=None):
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]+',tag):raise ValueError('Use a simple unique tag')
    report=r.HERE/'reports'/tag;report.mkdir(parents=True,exist_ok=False)
    hydro_report=r.HERE/'reports'/HYDRO_TAG
    seed=r.HERE/'runs'/f'{HYDRO_TAG}_fine_t8'
    reference=r.HERE/'runs'/DELFT_TAG
    runtime=r.HERE/'runs'/f'{RUNTIME_TAG}_runtime'
    def status(state,phase,**extra):
        r.write_json(report/'status.json',dict(status=state,phase=phase,pid=os.getpid(),
            updated_utc=datetime.now(timezone.utc).isoformat(),**extra))
    r.write_json(report/'plan.json',dict(hydraulic_report=str(hydro_report),seed=str(seed),
        runtime=str(runtime),delft_reference=str(reference),threads=8,dx_m=.10625,
        morfac=1,screen_end_s=120,comparison_end_s=3600,timeout_per_phase_s=86400,
        secondary_seed='Reset to zero in both engines, matching the preserved reference protocol',
        source_change='BRAIDS-only candidate curvature production regularization, ell=2h',
        delft_executed=False,delft_source_modified=False,production_promoted=False,
        runner_sha256=r.sha256(Path(__file__)),
        note='Screen and full run independently start from qualified hydraulics; screen is not appended. '
             'This repeats 120 s to keep the full trajectory contiguous and directly comparable. '
             'Same-resolution meshes have different topology. No equilibrium accuracy ranking.'))
    old=r.shared.verified_braids_build
    phase='hydraulic_wait'
    try:
        status('waiting',phase,supervisor_pid=after_pid)
        current=json.loads((hydro_report/'status.json').read_text())
        if current.get('status')=='running':
            if after_pid is None:raise RuntimeError('Hydraulics still running; supply its verified supervisor PID')
            wait_for_supervisor(after_pid)
        qualification=completed_hydraulics(hydro_report)
        r.write_json(report/'hydraulic_qualification.json',qualification)
        exe,identity=locked_candidate(runtime)
        r.write_json(report/'build.json',identity)
        # Re-read the seed and verify it against the completed qualification;
        # never trust stale status JSON when a run has been changed afterwards.
        hydro=a.load(seed)
        if r.sha256(seed/'execution.json')!=qualification['cases'][1]['execution_sha256']:
            raise ValueError('Qualified hydraulic execution changed')
        if r.sha256(r.shared.native_frames_path(seed))!=qualification['cases'][1]['frames_sha256']:
            raise ValueError('Qualified hydraulic frames changed')
        if hydro['execution']['executable_sha256']!=identity['executable_sha256']:
            raise ValueError('Warm-up used a different candidate runtime')
        coarse=a.load(r.HERE/'runs'/f'{HYDRO_TAG}_coarse_t4')
        if r.sha256(r.shared.native_frames_path(coarse['folder']))!=qualification['cases'][0]['frames_sha256']:
            raise ValueError('Qualified coarse hydraulic frames changed')
        mesh_screen=hydraulic_mesh_screen(coarse,hydro)
        r.write_json(report/'hydraulic_mesh_screen.json',mesh_screen)
        if not mesh_screen['passed']:raise RuntimeError('Bend-scale coarse/fine response needs review')
        delft=a.load(reference)
        if not np.isclose(delft['th'][-1],3600.):raise ValueError('Reference must reach 3600 s')
        r.write_json(report/'delft_reference.json',dict(folder=str(reference),
            input_sha256=r.sha256(reference/'input_provenance.json'),
            execution_sha256=r.sha256(reference/'execution.json'),
            trim_sha256=r.sha256(reference/'trim-basin.nc'),
            trih_sha256=r.sha256(reference/'trih-basin.nc'),metrics=a.audit(delft)))
        r.shared.verified_braids_build=lambda:(exe,identity)
        for phase,end,output in [('screen',120.,20.),('mobile',3600.,600.)]:
            folder=r.HERE/'runs'/f'{tag}_{phase}_braids'
            status('preparing',phase,folder=str(folder))
            r.prepare_native(folder,dx=.10625,end=end,output=output,dt=.0125,morfac=1,
                             hydraulic=False,seed=seed)
            p=json.loads((folder/'input_provenance.json').read_text())
            verify_common_configuration(p,delft['p'])
            p.update(engine_source_modified=True,production_source_modified=False,
                candidate='Unpromoted curvature-production regularization ell=2h',
                curvature_regularization_depth_factor=2.,
                hydraulic_qualification_sha256=r.sha256(hydro_report/'qualification.json'),
                hydraulic_seed_frames_sha256=r.sha256(r.shared.native_frames_path(seed)))
            r.write_json(folder/'input_provenance.json',p)
            # Verify executable identity again immediately before execution.
            locked_candidate(runtime)
            status('running',phase,folder=str(folder))
            execution=r.execute_native(folder,8,timeout_s=86400)
            data=a.load(folder);accounts=summary.evidence(data);metrics=accounts['metrics']
            gates=mobile_gates(metrics,accounts)
            r.write_json(report/f'{phase}_qualification.json',dict(gates=gates,**accounts))
            if phase=='screen':
                a.analyze(f'{tag}_screen',[folder])
                artifact_figure([data],r.HERE/'reports'/f'{tag}_screen')
                if not all(gates.values()):raise RuntimeError(f'Mobile screen failed: {gates}')
            else:
                a.analyze(tag,[folder,reference])
                artifact_figure([data,delft],report)
                r.write(report/'RESULTS.md',f'''# Regularized BRAIDS versus stock Delft3D, 3600 s

Both trajectories reached 3600 morphological seconds, MF=1, nominal horizontal
resolution 0.10625 m. Delft3D is the preserved unmodified one-layer run;
BRAIDS is the isolated secondary-flow regularization candidate. Mesh topology
and secondary-flow closure differ. This is not an exact replication of 2DEF.

Final operational/artifact screen: **{all(gates.values())}**. See
[individual checks](mobile_qualification.json). Small-scale bed noise and outlet
deposition still require figure review; passing these guards alone is not
experimental braids_support. No production/plugin promotion has been performed.

[Raw cell fields](fields.png) · [Experimental equilibrium profile comparison](profiles.png)
· [Cell-scale and outlet diagnostics](bed_scale_outlet.png)

The models are transient and the digitized experimental profiles are equilibrium
observations. Their instantaneous errors do not establish an accuracy ranking.
Next: review erosion/deposition location, cell-alternating changes, outlet export
and approach to the measured equilibrium shape before accepting the formulation.
''')
                status('complete',phase,folder=str(folder),operational_gates_passed=all(gates.values()),
                    experimental_validation_passed=False,production_promoted=False)
        return report
    except Exception as exc:
        status('failed',phase,error=f'{type(exc).__name__}: {exc}');raise
    finally:r.shared.verified_braids_build=old


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tag');parser.add_argument('--after-pid',type=int)
    args=parser.parse_args();run(args.tag,args.after_pid)
