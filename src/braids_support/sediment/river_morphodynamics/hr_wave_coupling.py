"""Isolated HR pressure-path correction in the conservative sediment bed row.

The sum of HR hydraulic fluctuations, including the two side pressures,
is [d(hu), d(hu*u)+d(Pstar), d(hu*v)], NOT d(F)+g*hbar*dz.
Keep physical h, momentum, z and sediment capacity; use the secant of
Pstar(eta)=g/2*max(eta-max(zL,zR),0)**2 in the pressure entries of A.
Then the hydraulic rows of A dU equal that path jump (Roe velocities).
This does not make the split sediment/hydraulic update a coupled Riemann
solver, nor establish entropy/positivity of the full morphodynamic scheme.
CPU/OpenMP and CUDA compile the same generated arithmetic. Research only.
"""
from pathlib import Path
from . import consistent_wave_state as traces
from .sediment_wave import WaveEngine
from braids_core.sediment.bend_methods_v1 import digest


PRESSURE = '''
  // The common bed determines ONLY the hydraulic opening. Never replace
  // physical z or physical depth in Exner/capacity by the common bed.
  double top=fmax(s[0][3],s[1][3]);
  double eta0=s[0][0]+s[0][3],eta1=s[1][0]+s[1][3];
  double hs0=fmax(0.,eta0-top),hs1=fmax(0.,eta1-top);
  // Stable pressure divided difference, including a dry/active crossing.
  double deta=eta1-eta0, opening;
  if(hs0>0. && hs1>0.)opening=.5*(hs0+hs1);
  else if(deta!=0.)opening=.5*(hs0+hs1)*(hs1-hs0)/deta;
  else opening=0.;
  double pressure_jump=.5*9.81*(hs0+hs1)*(hs1-hs0);
'''


def source(*,jump_only=False,cell_crest=False):
    code=traces.source()
    needle='  // Positive EH principal-part linearization'
    assert code.count(needle)==1
    code=code.replace(needle,PRESSURE+('' if jump_only else '  gh=9.81*opening;\n')+needle)
    old='s[1][0]*(s[1][1]*s[1][1]+.5*9.81*s[1][0])-s[0][0]*(s[0][1]*s[0][1]+.5*9.81*s[0][0])+gh*(s[1][3]-s[0][3])'
    assert code.count(old)==1
    code=code.replace(old,'s[1][0]*s[1][1]*s[1][1]-s[0][0]*s[0][1]*s[0][1]+pressure_jump')
    if cell_crest:code=code.replace('double top=fmax(s[0][3],s[1][3]);','double top=fmax(fmax(s[0][3],s[1][3]),fmax(z[o],z[j]));')
    return code


def install(a,folder,*,jump_only=False,cell_crest=False):
    if 'hydrostatic_step_probe' not in a.identity:
        raise ValueError('HR wave path requires the HR hydraulic interface')
    # Install fresh trace owner only after the final hydraulic library exists.
    traces.install(a,folder)
    p=a.physics
    if cell_crest and 'hr_cell_crest' not in a.identity:raise ValueError('Cell crest needs matching hydraulic geometry')
    from braids_core.windows_paths import filesystem_path
    target=filesystem_path(folder)/('hr_wave_crest.h' if cell_crest else ('hr_wave_jump.h' if jump_only else 'hr_wave_matched.h'))
    code=source(jump_only=jump_only,cell_crest=cell_crest)
    if target.exists():assert target.read_text()==code
    else:target.write_text(code)
    corrected=WaveEngine(a.backend,p.flow.threads,matrix=True,source=target)
    # traces' launch closure captures its original engine. Replace that
    # object's kernels/library, preserving its fresh-state launch adapter.
    engine=p.wave_engine
    engine.fn=corrected.fn
    if a.backend=='cpu':engine.lib=corrected.lib;engine.dirs=corrected.dirs
    else:engine.module=corrected.module
    engine.identity=corrected.identity
    p.identity['hr_wave_pressure_path']=dict(source_sha256=digest(target),
        driver_sha256=digest(__file__),runtime=corrected.identity,
        jump_only=jump_only,cell_crest=cell_crest,physical_capacity_and_bed_unchanged=True,
        pressure='clipped opening secant, with physical conserved-state Roe advection',
        public_support=False)
    return a
