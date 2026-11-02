"""Qualify the production-shaped dispatcher against the frozen pre-change build.

No research aborts: unqualified physics, small grids and AMR retain the original
implementation. Production adoption is a separate reviewed apply_patch step.
"""
from pathlib import Path
import argparse
from . import workflow_performance as p

TAG = 'workflow_production_20260930_v1'
VARIANTS = ('reference', 'optimized')

INITIALIZE = '''
 ! Qualified same-equation fixed-grid hydraulic workflow. Resolve after restart
 ! and inspect all forcing knots; dormant sediment inventories remain untouched.
 ! Keep the original implementation for coupled physics, AMR and small grids.
 hydraulic_workflow_fast=nc>=4096.and.scheme==6.and..not.amr_enabled.and. &
  bedlaw==0.and.exchange_mode==0.and..not.suspended_transport_active.and. &
  .not.secondary_flow.and..not.bank_failure.and..not.bank_avalanche.and. &
  .not.general_avalanche.and..not.bedform_roughness.and..not.bedload_adaptation.and. &
  bank_erosion_fraction==0.d0.and.bed_slope==0.and.horizontal_viscosity==0.d0.and. &
  nstate==5.and.nfrac==1.and.all(ws_fraction==0.d0).and. &
  all(forcing(3:4,:)==0.d0).and.all(series_concentration==0.d0).and. &
  all(series_bed_input==0.d0).and.all(fraction_concentration==0.d0).and. &
  all(fraction_bed_input==0.d0).and.all(fraction_series_concentration==0.d0).and. &
  all(fraction_series_bed_input==0.d0)
'''


def guarded_bookkeeping(euler):
    euler=p.m.replace_once(euler,'theta=1;stage_failure=0;stage_cfl_peak=0',
        'if(.not.hydraulic_workflow_fast) theta=1\n  stage_failure=0;stage_cfl_peak=0')
    start=euler.index('   do g=1,nfrac\n    bi=bed_index(g);outgoing=0')
    end=euler.index('   enddo',euler.index('theta(g,c)=',start))+len('   enddo')
    body=euler[start:end]
    euler=euler[:start]+'   if(.not.hydraulic_workflow_fast) then\n'+body+'\n   endif'+euler[end:]
    for marker in (
        '  !$omp parallel do schedule(static,256) if(parallel_faces) default(shared) &\n'+
        '  !$omp& private(work,f,c,n,b,g,bi,ci,donor,depth,velocity,secondary_equilibrium,secondary_high,secondary_active,valid)',
        '  !$omp parallel do schedule(static,256) if(ntarget*nfrac>=8192) default(shared) private(target,c,g,k,f,si,bi,sumflux)',
    ):
        start=euler.index(marker)
        end=euler.index('  !$omp end parallel do',start)+len('  !$omp end parallel do')
        body=euler[start:end]
        euler=euler[:start]+'''  ! With no bed flux or transported scalar, next=state already retains
  ! these inventories. Keep the complete original sweep for every other case.
  if(.not.hydraulic_workflow_fast) then
'''+body+'\n  endif'+euler[end:]
    return euler


def transform(original, variant):
    assert variant in VARIANTS
    if variant=='reference':return original
    text=p.m.replace_once(original,' logical :: suspended_transport_active=.true.',
        ' logical :: suspended_transport_active=.true.\n'+
        ' logical :: hydraulic_workflow_fast=.false.\n'+
        ' integer :: profile_hydraulic_workflow_calls=0')
    text=text.replace('\ncontains\n','\ncontains\n'+p.fused_routine(original),1)
    before=p.m.routine(text,'euler')
    after=p.m.replace_once(before,'''  call gradients(v,bc_concentration,grad,vlo,vhi)
  downslope=0
  if(bed_slope==2) call bed_downslopes(bed,downslope)
  if(scheme==6) call prepare_wall_traces(state,v,vlo,vhi)
  call prepare_wet_dry_traces(state,v,grad,vlo,vhi)''','''  if(hydraulic_workflow_fast) then
   downslope=0
   call prepare_hydraulic_traces_fused(state,v,bc_concentration,grad,vlo,vhi)
   if(profile_enabled) profile_hydraulic_workflow_calls=profile_hydraulic_workflow_calls+1
  else
   call gradients(v,bc_concentration,grad,vlo,vhi)
   downslope=0
   if(bed_slope==2) call bed_downslopes(bed,downslope)
   if(scheme==6) call prepare_wall_traces(state,v,vlo,vhi)
   call prepare_wet_dry_traces(state,v,grad,vlo,vhi)
  endif''')
    text=p.m.replace_once(text,before,guarded_bookkeeping(after))
    text=p.m.replace_once(text," call get_environment_variable('BRAIDS_ADAPTATION_DIAGNOSTICS'",
        INITIALIZE+" call get_environment_variable('BRAIDS_ADAPTATION_DIAGNOSTICS'")
    marker='  profile_regrid_calls=0;profile_prepare_calls=0;profile_euler_calls=0;profile_source_calls=0;profile_output_calls=0'
    text=p.m.replace_once(text,marker,marker+'\n  profile_hydraulic_workflow_calls=0')
    marker="  write(*,'(a,f12.4,a,i0)') 'profile: Euler update wall seconds=',profile_euler_s,' calls=',profile_euler_calls"
    text=p.m.replace_once(text,marker,marker+"\n  write(*,'(a,i0)') 'profile: optimized hydraulic workflow calls=',profile_hydraulic_workflow_calls")
    for name in ('wave_flux','advance','source','gradients','prepare_wall_traces','prepare_wet_dry_traces',
                 'face_value','face_elevations','prepare_hydraulic_stage'):
        assert p.m.routine(text,name)==p.m.routine(original,name),name
    return text


def paths():return p.paths(TAG)
def build():p.build(TAG,variants=VARIANTS,transformer=transform,builder=__file__)
def benchmark():p.benchmark(TAG,variants=VARIANTS)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('build','benchmark'))
    globals()[parser.parse_args().stage]()
