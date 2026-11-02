"""Isolated variable-step SSPMS(2) integration around the frozen f-wave Euler leg.

Input is the frozen larger_timesteps_20261001_v3/runtime_ssp43 main source.
Method 0 retains its complete SSP43 advance routine verbatim (renamed only).
Both controls and candidates receive the already audited stage-outlet limiter
bound extension. Neither source nor executable is a production replacement.
"""
from . import larger_timesteps as p

HERE = p.r.HERE

DECLARATIONS = """ integer :: explicit_history_method=0,implicit_outlet_extension=1
 integer :: explicit_history_count=0,explicit_history_position=1
 integer :: explicit_trial_reset_kind=0
 integer(kind=8) :: explicit_history_steps=0,explicit_history_startup_steps=0,explicit_history_failures=0
 integer(kind=8) :: explicit_history_invalid_span_attempts=0,explicit_history_friction_rejections=0
 integer(kind=8) :: explicit_history_knot_resets=0,explicit_history_growth_resets=0
 real(8),allocatable :: explicit_history_state(:,:,:),explicit_history_ledger(:,:,:),explicit_history_time(:)
 real(8),allocatable :: explicit_friction_rate(:)
 real(8) :: explicit_trial_span=0,explicit_trial_ratio=0,explicit_trial_friction_peak=0,explicit_trial_physical_cfl=0
 real(8) :: explicit_history_minimum_span=huge(1.d0),explicit_history_maximum_span=0,explicit_history_maximum_ratio=0
 real(8) :: explicit_history_maximum_friction=0,explicit_history_maximum_physical_cfl=0
 real(8) :: explicit_history_physical_time=0,explicit_history_startup_time=0,explicit_history_source_seconds=0
 real(8) :: explicit_last_accepted_dt=0
"""

SETTINGS = """
 call get_environment_variable('BRAIDS_EXPLICIT_HISTORY',profile_setting,status=environment_status)
 if(environment_status/=0) error stop 'explicit research requires a recorded history method'
 read(profile_setting,*,iostat=ios) explicit_history_method
 if(ios/=0) error stop 'invalid explicit history method'
 if(explicit_history_method/=0.and.explicit_history_method/=3.and.explicit_history_method/=4.and. &
    explicit_history_method/=6.and.explicit_history_method/=10) error stop 'history method must be 0, 3, 4, 6 or 10'
 call get_environment_variable('BRAIDS_EXPLICIT_DTMAX',profile_setting,status=environment_status)
 if(environment_status==0) then
  read(profile_setting,*,iostat=ios) dtmax
  if(ios/=0) error stop 'invalid research dtmax'
  if(.not.ieee_is_finite(dtmax).or.dtmax<=0) error stop 'invalid research dtmax'
  step_hint=dtmax
 endif
 call get_environment_variable('BRAIDS_RESEARCH_OUTLET_EXTENSION',profile_setting,status=environment_status)
 if(environment_status==0) then
  read(profile_setting,*,iostat=ios) implicit_outlet_extension
  if(ios/=0) error stop 'invalid outlet extension switch'
 endif
 if(implicit_outlet_extension/=0.and.implicit_outlet_extension/=1) error stop 'outlet extension switch must be 0 or 1'
 write(*,'(a,i0,a,f10.4,a,es12.4)') 'research: explicit history method=',explicit_history_method, &
  ' Euler CFL=',cfl_target,' dtmax=',dtmax
 write(*,'(a,i0)') 'research: outlet bound extension=',implicit_outlet_extension
"""

REPORT = """
  write(*,'(a,i0)') 'explicit history: method=',explicit_history_method
  write(*,'(a,i0)') 'explicit history: accepted multistep steps=',explicit_history_steps
  write(*,'(a,i0)') 'explicit history: SSP43 startup/fallback steps=',explicit_history_startup_steps
  write(*,'(a,i0)') 'explicit history: failed attempts=',explicit_history_failures
  write(*,'(a,i0)') 'explicit history: invalid span fallback attempts=',explicit_history_invalid_span_attempts
  write(*,'(a,i0)') 'explicit history: friction rejected attempts=',explicit_history_friction_rejections
  write(*,'(a,i0)') 'explicit history: forcing knot resets=',explicit_history_knot_resets
  write(*,'(a,i0)') 'explicit history: excessive step growth resets=',explicit_history_growth_resets
  write(*,'(a,es22.14)') 'explicit history: multistep physical seconds=',explicit_history_physical_time
  write(*,'(a,es22.14)') 'explicit history: startup/fallback physical seconds=',explicit_history_startup_time
  write(*,'(a,es22.14)') 'explicit history: source/history combination wall seconds=',explicit_history_source_seconds
  write(*,'(a,es22.14)') 'explicit history: minimum accepted history span=', &
   merge(explicit_history_minimum_span,0.d0,explicit_history_steps>0)
  write(*,'(a,es22.14)') 'explicit history: maximum accepted history span=',explicit_history_maximum_span
  write(*,'(a,es22.14)') 'explicit history: maximum accepted dt/span=',explicit_history_maximum_ratio
  write(*,'(a,es22.14)') 'explicit history: maximum accepted friction FE factor=',explicit_history_maximum_friction
  write(*,'(a,es22.14)') 'explicit history: maximum accepted physical CFL=',explicit_history_maximum_physical_cfl
"""


def transform(original):
    """Return a same-binary baseline/candidate source without touching disk."""
    text = p.replace(original, ' implicit none\n integer :: nc,nf',
                     ' implicit none\n'+DECLARATIONS+' integer :: nc,nf')
    advance = p.methods.routine(text, 'advance')
    explicit = advance.replace('subroutine advance(', 'subroutine research_explicit_advance(', 1)
    history = advance.replace('subroutine advance(', 'subroutine research_history_advance(', 1)
    history = p.replace(history, '  integer :: target',
                        '  integer :: target\n  logical :: used_history')
    # Stage preparation clears dry momentum and caps velocity. Use the already
    # declared scratch state so even this preparation cannot mutate accepted q
    # on a rejected attempt (notably a cold/restart state at the dry threshold).
    history = p.replace(history, '  call prepare_hydraulic_stage(q)',
                        '  !$omp parallel do schedule(static,256) default(shared) private(target)\n'
                        '  do target=1,nc\n'
                        '   a(:,target)=q(:,target)\n'
                        '  enddo\n'
                        '  !$omp end parallel do\n'
                        '  call prepare_hydraulic_stage(a)')
    history = p.replace(history,
                        '  call research_ssp43(q,dt,step_morfac,c,fa,research_momentum,ok)',
                        '  call research_history_step(a,dt,step_morfac,c,fa,research_momentum,used_history,ok)')
    history = p.replace(history, '  call source(c,dt)', '  if(.not.used_history) call source(c,dt)')
    # Retain baseline source accounting for startup, avoid charging a spurious
    # post-combination source to true LMM steps (counted inside the new helper).
    history = p.replace(history,
                        '  if(profile_enabled) then\n   profile_source_s=profile_source_s+omp_get_wtime()-profile_start;profile_source_calls=profile_source_calls+1',
                        '  if(profile_enabled.and..not.used_history) then\n   profile_source_s=profile_source_s+omp_get_wtime()-profile_start;profile_source_calls=profile_source_calls+1')
    # All acceptance checks precede this insertion. Rejected attempts must not
    # advance the ring, including failures after a successful Euler leg.
    history = p.replace(history,
                        '  !$omp parallel do schedule(static,256) if(nc*nstate>=131072) default(shared) private(target)',
                        '  call explicit_history_commit(a,ledger,time,dt,used_history)\n'
                        '  !$omp parallel do schedule(static,256) default(shared) private(target)')
    text = p.replace(text, advance, explicit+'\n\n'+history)
    extension = (HERE/'native/implicit_outlet_bounds.inc').read_text()
    helper = (HERE/'native/explicit_multistep.inc').read_text()
    text = p.replace(text, ' subroutine profile_report()',
                     extension+'\n'+helper+'\n subroutine profile_report()')
    for name in ('gradients', 'prepare_hydraulic_traces_fused'):
        before = p.methods.routine(text, name)
        after = p.replace(before, '   theta=1', '   call implicit_outlet_bounds(c,v,grad,vlo,vhi)\n   theta=1')
        text = p.replace(text, before, after)
    marker = ' ! Qualified same-equation fixed-grid hydraulic workflow. Resolve after restart'
    text = p.replace(text, marker, p.methods.GUARD+SETTINGS+marker)
    report = p.methods.routine(text, 'profile_report')
    text = p.replace(text, report, p.replace(report, '  if(.not.profile_enabled) return',
                                            '  if(.not.profile_enabled) return'+REPORT))
    # Augmented flux wrapper, complete limiter/flux/update operator, spatial
    # reconstruction formulae, closures and original source remain immutable.
    changed = {'advance', 'gradients', 'prepare_hydraulic_traces_fused', 'profile_report'}
    import re
    for name in re.findall(r'(?im)^ *(?:subroutine|function) (\w+)\(', original):
        if name not in changed:
            assert p.methods.routine(text, name) == p.methods.routine(original, name), name
    assert p.methods.routine(text, 'research_explicit_advance').replace(
        'subroutine research_explicit_advance(', 'subroutine advance(', 1) == advance
    return text
