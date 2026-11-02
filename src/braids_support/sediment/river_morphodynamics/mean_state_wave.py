"""Research correction: evaluate frozen EH wave coefficient at its mean state.

Arithmetic mean endpoint capacity is not capacity at the Roe velocity/depth,
especially with dwnvel's shallow log-law conversion. No physical capacity,
slope, reconstruction, hydraulic flux or conservative assembly is changed.
The depth dependence of roughness/shear remains frozen in this positive
principal-part Jacobian, as before; this is not the full nonlinear Jacobian.
"""
from pathlib import Path
from . import hr_wave_coupling as hr
from .sediment_wave import WaveEngine
from braids_core.sediment.bend_methods_v1 import digest


def source():
    code=hr.source()
    code=code.replace('double*diagnostic){','double*diagnostic,const double*rough,double diameter){')
    old='double mag=.5*(hypot(cell[3*n+o],cell[4*n+o])+hypot(cell[3*n+j],cell[4*n+j]))/2650.;'
    assert code.count(old)==1
    return code.replace(old,'''// Evaluate the frozen constitutive coefficient at the SAME mean h,u,v.
  // Working roughness depth is floored; characteristic elevation is physical.
  // Endpoint capacity (a fifth power) cannot serve as a mean-state magnitude.
  double zrough=.5*(rough[2*n+o]+rough[2*n+j]);
  double cmean=sqrt(9.81)/.41*log1p(fmax(.01,h)/fmax(1.e-8,exp(1.)*zrough));
  double velocity_factor=.41*cmean/(sqrt(9.81)*log1p(h*double(float(.368))/zrough));
  double cwork=fmax(20.,cmean),ve=sqrt(sp2)*velocity_factor;
  double mag=.05*pow(ve,5)/(sqrt(9.81)*pow(cwork,3)*1.65*1.65*diameter);''')


def install(a,folder):
    p=a.physics
    if p.time or p.steps or p._open:raise ValueError('Install before stepping')
    if 'hr_wave_pressure_path' not in p.identity:raise ValueError('Requires matched fresh HR wave states')
    from braids_core.windows_paths import filesystem_path
    target=filesystem_path(folder)/'mean_state_wave.h';target.parent.mkdir(parents=True,exist_ok=True)
    from braids_core.sediment import class_wave_cache
    engine=p.wave_engine;cached=getattr(engine,'_class_wave_cache_dispatch',False)
    code=class_wave_cache.source(source()) if cached else source()
    if target.exists():assert target.read_text()==code
    else:target.write_text(code)
    corrected=WaveEngine(a.backend,p.flow.threads,matrix=True,source=target,
        names=class_wave_cache.NAMES if cached else ('wave_state_gradients','matrix_bed_faces'))
    old=engine.launch
    engine.fn=corrected.fn
    if cached:
        class_wave_cache.bind(engine)
        class_wave_cache.enable(engine,a.xp,p.nf)
    if a.backend=='cpu':engine.lib=corrected.lib;engine.dirs=corrected.dirs
    else:engine.module=corrected.module
    engine.identity=corrected.identity
    def launch(name,count,*args):
        if name=='matrix_bed_faces':args=(*args,p.effective_fields,float(p.parameters.diameter))
        return old(name,count,*args)
    engine.launch=launch
    p.identity['mean_state_wave']=dict(wrapper_sha256=digest(__file__),runtime=corrected.identity,
        class_cache_sha256=digest(class_wave_cache.__file__),
        class_cache=('same-call class-independent mean hydraulics and roughness only' if cached else 'disabled: historical trace launch ABI'),
        coefficient='EH magnitude at Roe velocity, arithmetic physical depth and mean z0; frozen coefficient derivative',
        physical_capacity_changed=False,public_support=False)
    return a
