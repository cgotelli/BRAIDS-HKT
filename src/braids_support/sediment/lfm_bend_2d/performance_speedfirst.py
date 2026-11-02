"""Speed-first auxiliary-method experiments, with the GeoClaw core frozen.

Each experiment freezes all sources. No Delft3D source/runtime is changed.
The baseline is the *latest* auxiliary-only MIC runtime, not the older Jacobi.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from . import run_case as r
from . import optimize_performance as old
from . import performance_auxiliary as aux
from . import performance_methods as methods
from . import profile_current as audit

TAG = 'performance_speedfirst_20260930_v1'
BASE = r.HERE / 'runs/performance_auxiliary_20260930_v1'
LIMITS = dict(bed_normalized_rms_percent=10., bed_peak_error_percent=20.,
              depth_normalized_rms_percent=5., velocity_normalized_rms_percent=10.,
              bed_pattern_agreement_percent=90., sediment_export_difference_percent=10.,
              water_increment_budget_m3=1e-8, sediment_increment_budget_kg=1e-6)
VARIANTS = ('muscl', 'filter6', 'filter4', 'combined6', 'combined4', 'parallel6', 'persistent6', 'parallel4', 'persistent4')


def muscl_source(text):
    before = methods.routine(text, 'bedload_capacity_face')
    after = before.replace(
        '  ! WENO-Z is used on certified five-cell Cartesian stencils.  Cut faces use\n'
        '  ! the bounded least-squares gradient above.  Both traces are finally bounded\n',
        '  ! Second-order bounded least-squares (MUSCL) sediment reconstruction on\n'
        '  ! both regular and cut cells. Hydraulic WENO is unchanged. Traces are bounded\n')
    start = after.index('  if(weno_cells(1,side,f)>0) then')
    stop = after.index('\n  if(present(consistent_stage)) then', start)
    after = after[:start] + '''  r=face_displacement(:,side,f)
  do m=1,2
   value(m)=value(m)+dot_product(gradient(m,:,g,c),r)
  enddo''' + after[stop:]
    after = after.replace('value(2),stencil(5),r(2)', 'value(2),r(2)').replace('side,k,m,n', 'side,m,n')
    assert after != before and 'reconstruct(stencil' not in after
    return audit.replace_once(text, before, after)


def build(root, variant):
    runtime = root / ('runtime_' + variant); runtime.mkdir(parents=True, exist_ok=False)
    parent = BASE / 'runtime_serial'; prior = json.loads((parent / 'build.json').read_text())
    command = prior['command'][:6]  # compiler and the recorded optimization/OpenMP flags
    assert command[-1] == '-ffixed-line-length-none'
    hashes = {}; guard = {}; tolerance = 1e-13
    for name, expected in prior['source_sha256'].items():
        source = parent / Path(name).name
        assert r.sha256(source) == expected
        text = source.read_text()
        if source.name == 'cutcell_solver.f90':
            original = text
            if variant == 'muscl' or variant.startswith('combined'): text = muscl_source(text)
            guard = aux.hydraulic_guard(text, original)
        elif source.name == 'secondary_curvature_regularization.f90' and variant != 'muscl':
            if variant.startswith(('parallel','persistent')):
                text = aux.PARALLEL.read_text()
                if variant.startswith('persistent'):
                    from .persistent_curvature_candidate import persistent_source
                    text = persistent_source(text)
            digits = 4 if variant.endswith('4') else 6; tolerance = 10. ** -digits
            norm = 'inner(area,area)' if variant.startswith(('parallel','persistent')) else 'sum(area*area)'
            text = audit.replace_once(text, f'tolerance=1.d-13*scale+1.d-16*sqrt({norm})',
                                      f'tolerance=1.d-{digits}*scale+1.d-{digits+6}*sqrt({norm})')
            text = text.replace('! The stronger preconditioner makes a stricter solve affordable. It also\n'
                                '  ! reduces sensitivity of nonlinear limiters to iterative-solve roundoff.',
                                '! Speed-first candidate: solve the same operator to a practical residual.\n'
                                '  ! Acceptance is based on bed patterns and conservation, not bitwise matching.')
        target = runtime / source.name; target.write_text(text)
        hashes[name] = r.sha256(target); command.append(str(target))
    executable = runtime / 'xbraids_candidate.exe'; command += ['-o', str(executable)]
    begin = time.perf_counter()
    proc = subprocess.run(command, cwd=runtime, capture_output=True, text=True, timeout=180)
    (runtime / 'build.log').write_text(proc.stdout + proc.stderr); proc.check_returncode()
    r.write_json(runtime / 'build.json', dict(command=command, source_sha256=hashes,
        executable_sha256=r.sha256(executable), baseline=str(parent), baseline_sha256=prior['executable_sha256'],
        variant=variant, depth_factor=2., relative_tolerance=tolerance, constant_mf_limit=25, series_mf_limit=20,
        hydrodynamic_routines_unchanged=guard, delft_modified=False, compile_wall_s=time.perf_counter()-begin))
    print('BUILT ' + variant, flush=True)


def exe(root, variant):
    runtime = BASE / 'runtime_serial' if variant == 'baseline' else root / ('runtime_' + variant)
    identity = json.loads((runtime / 'build.json').read_text()); path = runtime / 'xbraids_candidate.exe'
    assert r.sha256(path) == identity['executable_sha256']
    for source, expected in identity['source_sha256'].items(): assert r.sha256(runtime / Path(source).name) == expected
    return path


def compare(left, right):
    out = old.compare(left, right); out.pop('passed')
    lf = r.shared.read_frames(r.shared.native_frames_path(left))
    rf = r.shared.read_frames(r.shared.native_frames_path(right))
    lq = lf[-1]['q']; rq = rf[-1]['q']; area = np.load(left / 'analysis_geometry.npz')['area']
    def rms(x): return float(np.sqrt(np.dot(area, np.sum(x*x,axis=1) if x.ndim == 2 else x*x) / area.sum()))
    lb = lq[:,5:9].sum(axis=1)/r.BULK-r.RESERVOIR
    rb = rq[:,5:9].sum(axis=1)/r.BULK-r.RESERVOIR
    delta = rb-lb; active = abs(lb) > .002  # do not attach relative error to an almost-zero bed change
    lu = lq[:,1:3]/np.maximum(lq[:,:1],1e-8); ru = rq[:,1:3]/np.maximum(rq[:,:1],1e-8)
    out.update(bed_normalized_rms_percent=100*rms(delta)/max(rms(lb),.002),
        bed_peak_error_percent=100*float(np.max(abs(delta)))/max(float(np.max(abs(lb))),.005),
        depth_normalized_rms_percent=100*rms(rq[:,0]-lq[:,0])/max(rms(lq[:,0]),.001),
        velocity_normalized_rms_percent=100*rms(ru-lu)/max(rms(lu),.001),
        bed_pattern_agreement_percent=100*float(np.sum(area[active]*(np.sign(lb[active])==np.sign(rb[active])))/np.sum(area[active])) if np.any(active) else 100.,
        minimum_sediment_inventory_kg_m2=float(rq[:,3:9].min()),
        bed_difference_rms_m=rms(delta))
    initial=lf[0]['q'][:,5:9].sum(axis=1)/r.BULK-r.RESERVOIR
    out['bed_increment_normalized_rms_percent']=100*rms(delta)/max(rms(lb-initial),.0001)
    exports=[]
    for folder in (left,right):
        raw=audit.read_ledger(folder);nb,nledger=map(int,raw[:2]);records=raw[2:].reshape(-1,1+nb*nledger)
        change=(records[-1,1:]-records[0,1:]).reshape(nb,nledger)
        exports.append(float(np.maximum(-change[:,1:].sum(axis=1),0).sum()))
    out['sediment_export_difference_percent']=100*abs(exports[1]-exports[0])/max(exports[0],1e-8)
    out['gates']={k:(out[k]>=limit if k=='bed_pattern_agreement_percent' else out[k]<=limit) for k,limit in LIMITS.items()}
    out['gates'].update(finite=out['all_finite'],positive_water=out['minimum_depth_m']>0,
                        nonnegative_sediment=out['minimum_sediment_inventory_kg_m2']>=0)
    out['passed']=all(out['gates'].values())
    return out


def run(tag, stage, variant):
    root=r.HERE/'runs'/tag;report=r.HERE/'reports'/tag
    if stage=='build':
        root.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
        r.write_json(report/'plan.json',dict(baseline=str(BASE/'runtime_serial'),limits=LIMITS,
            rationale='Speed first: percent-level global morphology gates; strict conservation, positivity and stability.',
            bed_increment_error='Reported separately; short continuation does not qualify equilibrium morphology.',
            variants=list(VARIANTS),fixed=['GeoClaw hydraulics','SSPRK3','MF25','mesh','physics and boundaries'],
            parallel_policy='Re-measure parallel filter at practical tolerance; preserve parallel sediment cell/face work.',
            plugin_promotion=False,delft_modified=False))
        for name in (VARIANTS if variant=='all' else (variant,)): build(root,name)
        return
    if stage=='screen':
        rows=[];checks={}
        for size,source,start,duration,checkpoint,threads in aux.CASES:
            for name in ('baseline','muscl','filter6','filter4','combined6','combined4'):
                label=f'screen_{size}_{name}'
                rows.append(old.execute(root,label,exe(root,name),source,threads,duration,start,checkpoint))
                if name!='baseline': checks[f'{size}_{name}']=compare(root/f'screen_{size}_baseline',root/label)
                r.write_json(report/'screen.json',dict(rows=rows,checks=checks))
        return
    if stage=='benchmark':
        rows=[];checks={}
        orders = (('baseline','filter4','combined4'),('combined4','filter4','baseline'),
                  ('filter4','baseline','combined4')) if variant=='all' else (('baseline',variant),(variant,'baseline'))
        for size,source,start,duration,checkpoint,threads in aux.CASES:
            for repeat,order in enumerate(orders,1):
                for name in order:
                    label=f'{size}_r{repeat}_{name}'
                    rows.append(old.execute(root,label,exe(root,name),source,threads,2*duration,start,checkpoint))
                for name in order:
                    if name!='baseline': checks[f'{size}_r{repeat}_{name}']=compare(root/f'{size}_r{repeat}_baseline',root/f'{size}_r{repeat}_{name}')
                r.write_json(report/'benchmark.json',dict(rows=rows,checks=checks,variant=variant))
                assert all(x['passed'] for x in checks.values()),checks
        return
    if stage=='regression':
        row=old.execute(root,'long_'+variant,exe(root,variant),audit.SOURCE,8,24,2880,audit.SOURCE/'checkpoint.bin')
        check=compare(BASE/'long_serial',root/('long_'+variant))
        r.write_json(report/('regression_'+variant+'.json'),dict(row=row,check=check,reference=str(BASE/'long_serial')))
        print(json.dumps(check),flush=True)
        assert check['passed'],check
        return
    if stage=='reproducibility':
        rows=[]
        for label,threads,duration,start,checkpoint in [
            ('large_t1',1,.2,0,None),('large_split_a',16,.1,0,None),
            ('large_split_b',16,.1,.1,root/'large_split_a/checkpoint.bin')]:
            rows.append(old.execute(root,label,exe(root,variant),aux.CASES[1][1],threads,duration,start,checkpoint))
        ref=r.sha256(root/f'large_r1_{variant}/checkpoint.bin')
        checks={name:r.sha256(root/name/'checkpoint.bin')==ref for name in ('large_t1','large_split_b')}
        r.write_json(report/'reproducibility.json',dict(rows=rows,checks=checks,variant=variant))
        assert all(checks.values()),checks
        return
    raise ValueError(stage)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['build','screen','benchmark','regression','reproducibility'])
    p.add_argument('--tag',default=TAG);p.add_argument('--variant',default='all',choices=['all',*VARIANTS])
    args=p.parse_args();run(args.tag,args.stage,args.variant)
