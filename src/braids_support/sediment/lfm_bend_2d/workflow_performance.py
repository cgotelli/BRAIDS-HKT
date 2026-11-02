"""Same-equation workflow experiments with an immutable GeoClaw reference."""
from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

from . import rectangular_performance as p

h,m=p.h,p.m
TAG='workflow_optimization_20260930_v1'
VARIANTS=('reference','fused')


def fused_routine(original):
    """Reuse each original cell body verbatim (apart from one local rename).

    A one-iteration DO preserves original outer-cell CYCLE statements while
    allowing the next preparation phase to execute for the same cell. Every
    neighbor read is from immutable primitive v; intermediate results are local
    to c. This is not a numerical reconstruction change.
    """
    grad=m.routine(original,'gradients')
    grad=grad[grad.index('   if(.not.active_cell(c)) cycle'):grad.rindex('  enddo\n  !$omp end parallel do')]
    grad=re.sub(r'\bn\b','ncell',grad)
    wall=m.routine(original,'prepare_wall_traces')
    wall=wall[wall.index('   if(state(1,c)<=dry_tolerance) cycle'):wall.rindex('  enddo\n  !$omp end parallel do')]
    wet=m.routine(original,'prepare_wet_dry_traces')
    wet=wet[wet.index('   h=state(1,c);flatten='):wet.rindex('  enddo\n  !$omp end parallel do')]
    return '''
 subroutine prepare_hydraulic_traces_fused(state,v,bc_concentration,grad,vlo,vhi)
  ! Same fixed-grid scheme-6 formulas and per-cell accumulation order.
  ! Only immutable v/geometry cross cell boundaries in these three kernels.
  real(8), intent(in) :: state(nstate,nc),v(nprimitive,nc),bc_concentration(nfrac,nb)
  real(8), intent(out) :: grad(nprimitive,2,nc),vlo(nprimitive,nc),vhi(nprimitive,nc)
  real(8) :: a11,a12,a22,b(nprimitive,2),r(2),w,det,delta(nprimitive),theta(nprimitive),z,adjacent(nprimitive)
  real(8) :: stencil(5,nprimitive),tangential(5),n(2),tangent(2),un,coef,tlo,thi,ut,h,point(2),ratio,normal_q,eta,zb
  integer :: c,k,f,ncell,m,g,once,a,side,slot,kk,s,kind,other
  logical :: wet,flatten
  !$omp parallel do schedule(static,256) default(shared) &
  !$omp& private(c,k,f,ncell,m,g,once,a,side,slot,kk,s,kind,other,a11,a12,a22,b,r,w,det,delta,theta,z,adjacent,stencil,tangential,n,tangent,un,coef,tlo,thi,ut,h,point,ratio,normal_q,eta,zb,wet,flatten)
  do c=1,nc
   grad(:,:,c)=0;vlo(:,c)=v(:,c);vhi(:,c)=v(:,c)
   wall_ready(:,c)=.false.;wall_values(:,:,c)=0;hydro_constant(c)=.false.
   do once=1,1
'''+grad+'''   enddo
   do once=1,1
'''+wall+'''   enddo
'''+wet+'''  enddo
  !$omp end parallel do
 end subroutine
'''


def transform(original,variant):
    assert variant in VARIANTS
    if variant=='reference':return original
    text=m.replace_once(original,' call load_case(trim(deck))',' call load_case(trim(deck))'+m.GUARD+
                        " if(scheme/=6) error stop 'fused research build requires scheme 6'\n")
    text=text.replace('\ncontains\n','\ncontains\n'+fused_routine(original),1)
    before=m.routine(text,'euler')
    after=m.replace_once(before,'''  call gradients(v,bc_concentration,grad,vlo,vhi)
  downslope=0
  if(bed_slope==2) call bed_downslopes(bed,downslope)
  if(scheme==6) call prepare_wall_traces(state,v,vlo,vhi)
  call prepare_wet_dry_traces(state,v,grad,vlo,vhi)''','''  downslope=0
  if(bed_slope==2) call bed_downslopes(bed,downslope)
  call prepare_hydraulic_traces_fused(state,v,bc_concentration,grad,vlo,vhi)''')
    text=m.replace_once(text,before,after)
    for name in re.findall(r'(?im)^ *(?:subroutine|function) (\w+)\(',original):
        if name!='euler':assert m.routine(original,name)==m.routine(text,name),name
    return text


def paths(tag=TAG):return p.paths(tag)


def build(tag=TAG,*,variants=VARIANTS,transformer=transform,builder=None):
    builder=Path(builder) if builder else Path(__file__)
    root,report=paths(tag);root.mkdir(exist_ok=False);report.mkdir(exist_ok=False)
    _,parent=h.runtime(p.PARENT,'reference')
    shutil.copytree(p.PARENT/'runtime_reference',root/'runtime_reference')
    for variant in variants[1:]:
        folder=root/('runtime_'+variant);folder.mkdir();hashes={}
        for name in h.SOURCES:
            source=p.PARENT/'runtime_reference'/Path(name).name;target=folder/source.name
            assert h.r.sha256(source)==parent['source_sha256'][name]
            if name==h.SOURCES[-1]:
                old=source.read_text();new=transformer(old,variant);target.write_text(new)
                (folder/'candidate.patch').write_text(''.join(difflib.unified_diff(
                    old.splitlines(True),new.splitlines(True),fromfile='reference',tofile=variant)))
            else:shutil.copy2(source,target)
            hashes[name]=h.r.sha256(target)
        command=[parent['command'][0],'-O3','-flto=8','-fopenmp','-ffree-line-length-none','-ffixed-line-length-none',
                 *[str(folder/Path(s).name) for s in h.SOURCES],'-o',str(folder/'xbraids_candidate.exe')]
        start=time.perf_counter();result=subprocess.run(command,cwd=folder,capture_output=True,text=True,timeout=240)
        (folder/'build.log').write_text(result.stdout+result.stderr);result.check_returncode()
        h.r.write_json(folder/'build.json',dict(command=command,source_sha256=hashes,
            executable_sha256=h.r.sha256(folder/'xbraids_candidate.exe'),variant=variant,
            parent_executable_sha256=parent['executable_sha256'],compile_wall_s=time.perf_counter()-start,
            numerical_formulas_unchanged=True,builder_sha256=h.r.sha256(builder)))
        shutil.copy2(builder,folder/'builder_snapshot.py')
        if builder!=Path(__file__):shutil.copy2(Path(__file__),folder/'parent_builder_snapshot.py')
        print('BUILT '+variant,flush=True)


def benchmark(tag=TAG,*,variants=VARIANTS):
    root,report=paths(tag);old,_=p.paths();rows=[]
    cases=dict(small=(old/'input_small_reference',8),large=(old/'input_large_reference',32),
               bend=(p.PARENT/'input40_reference',8))
    for case,(source,threads) in cases.items():
        for repeat in range(3):
            for variant in variants[repeat%len(variants):]+variants[:repeat%len(variants)]:
                row=h.execute_native(root,source,f'bench_{case}_{variant}_r{repeat+1}',threads,variant)
                rows.append(dict(row,case=case,repeat=repeat+1));h.r.write_json(report/'benchmark.json',rows)


def stock(tag=TAG):
    root,report=paths(tag);old,_=p.paths();rows=[]
    sources=dict(small=old/'input_small_delft_matched',large=old/'input_large_delft_matched',
        bend=h.r.HERE/'runs'/'performance_gap_audit_20260930_v1'/'input_delft_matched_steps')
    for case,source in sources.items():
        for repeat in range(3):
            row=h.execute_delft(root,source,f'stock_{case}_r{repeat+1}',4)
            row['timers']=h.delft_timers(root/row['label'])
            rows.append(dict(row,case=case,repeat=repeat+1));h.r.write_json(report/'stock.json',rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('build','benchmark','stock'))
    parser.add_argument('--tag',default=TAG);args=parser.parse_args();globals()[args.stage](args.tag)
