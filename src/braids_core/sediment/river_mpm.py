"""MPM/Egiazaroff physical bundle on the authoritative river layer owner."""
from pathlib import Path
from types import MethodType
import numpy as np
from .flow2dh_eh_operator import ClassicFlowEH
from .bend_methods_v1 import digest
from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine
from braids_support.sediment.river_morphodynamics import mean_state_wave


class RiverMPM(ClassicFlowEH):
    def __init__(self,backend,*,diameter,chezy,threads=4):
        super().__init__(backend,diameter=diameter,chezy=chezy,threads=threads)
        self.engine=WaveEngine(backend,threads,source=Path(__file__).with_suffix('.h'),names=('mixture','mpm'))
        self.d90=None;self.hiding=None
        self.identity=dict(backend=backend,diameter_m=diameter,chezy=chezy,runtime=self.engine.identity,
            wrapper_sha256=digest(__file__),method='FLOW_tranb2_MPM',critical_shields=.047,
            shear='resolved_Chezy_and_mixture_D90_ripple_factor',hiding='critical_Shields_once',
            depth='FLOW erosed 1cm working-depth floor',slope='inherited Koch-Flokstra',public_support=False)

    def evaluate(self,q,hi,*,capacity_q=None,chezy=None):
        xp=self.xp;cq=q if capacity_q is None else capacity_q;n=q.shape[1]
        c=xp.full(n,self.c) if chezy is None else chezy
        for v,shape in ((q,(3,n)),(cq,(3,n)),(c,(n,)),(hi,(n,)),(self.d90,(n,)),(self.hiding,(n,))):
            if (not isinstance(v,xp.ndarray) or v.shape!=shape or v.dtype!=xp.float64
                    or not v.flags.c_contiguous or not bool(xp.all(xp.isfinite(v)))):
                raise ValueError('MPM requires finite contiguous resident FP64 inputs')
        if bool(xp.any(q[0]<=0)) or bool(xp.any(cq[0]<=0)) or bool(xp.any(c<20)) or bool(xp.any(self.d90<=0)) or bool(xp.any(self.hiding<=0)):
            raise ValueError('Invalid MPM depth, roughness or mixture')
        out=xp.zeros((6,n))
        self.engine.launch('mpm',n,n,self.d,q,cq,c,hi,self.d90,self.hiding,out)
        if not bool(xp.all(xp.isfinite(out))):raise ValueError('Nonfinite MPM output')
        return out


def wave_source():
    code=mean_state_wave.source()
    code=code.replace('const double*rough,double diameter){','const double*hide,const double*d90,const double*rough,double diameter){')
    old='double mag=.05*pow(ve,5)/(sqrt(9.81)*pow(cwork,3)*1.65*1.65*diameter);'
    assert code.count(old)==1
    code=code.replace(old,'''double dm90=.5*(d90[o]+d90[j]);
  double grain=18.*log10(12.*fmax(fmax(.01,h)/dm90,1.));
  double mu=fmin(1.,pow(cwork/grain,1.5));
  double beta=mu*velocity_factor*velocity_factor/(cwork*cwork*1.65*diameter);
  double excess=fmax(0.,beta*sp2-.047*.5*(hide[o]+hide[j]));
  double pref=8.*sqrt(9.81*1.65*diameter)*diameter;
  double mag=pref*excess*sqrt(excess);''')
    old='der[0]=-5.*factor*u;der[1]=factor*(1.+4.*u*u/sp2);der[2]=factor*4.*u*v/sp2;'
    assert code.count(old)==1
    return code.replace(old,'''// Finite at the mobility threshold: do not divide by excess.
  double radial=acceleration*3.*pref*sqrt(excess)*beta*sqrt(sp2)/h;
  der[0]=-radial*u;der[1]=factor+(radial-factor)*u*u/sp2;
  der[2]=(radial-factor)*u*v/sp2;''')


def install_mpm(a,folder,*,hiding='egiazaroff'):
    if a.time or a.steps or a.physics._open:raise ValueError('Install physical bundle before execution')
    if hasattr(a,'mpm_installed'):raise ValueError('MPM already installed')
    if hiding not in ('none','egiazaroff'):raise ValueError('This route admits none/Egiazaroff only')
    ng=len(a.methods.diameter_m)
    if not 1<=ng<=32:raise ValueError('MPM bundle admits 1..32 fractions')
    p=a.physics;xp=a.xp
    a.capacity=[RiverMPM(a.backend,diameter=d,chezy=p.chezy,threads=p.flow.threads) for d in a.methods.diameter_m]
    p.mpm_hiding=xp.ones((ng,p.n));p.mpm_d90=xp.zeros(p.n);p.mpm_mean=xp.zeros(p.n)
    diam=xp.asarray(a.methods.diameter_m);engine=a.capacity[0].engine
    original=a.bed_faces
    def faces(owner,q,z,hi,water=None):
        engine.launch('mixture',p.n,p.n,ng,int(hiding!='none'),owner.columns.available,diam,p.mpm_hiding,p.mpm_d90,p.mpm_mean)
        for k,cap in enumerate(owner.capacity):cap.hiding=p.mpm_hiding[k];cap.d90=p.mpm_d90
        return original(q,z,hi,water)
    a.bed_faces=MethodType(faces,a);a.mpm_installed=True
    from . import class_wave_cache
    wave=p.wave_engine;cached=hasattr(wave,'_class_shared')
    from braids_core.windows_paths import filesystem_path
    target=filesystem_path(folder)/'mpm_wave.h';target.parent.mkdir(parents=True,exist_ok=True);code=wave_source()
    if cached:code=class_wave_cache.source(code)
    if target.exists():assert target.read_text()==code
    else:target.write_text(code)
    corrected=WaveEngine(a.backend,p.flow.threads,source=target,
        names=class_wave_cache.NAMES if cached else ('wave_state_gradients','matrix_bed_faces'))
    wave=p.wave_engine;old=wave.launch;wave.fn=corrected.fn
    if cached:class_wave_cache.bind(wave)
    if a.backend=='cpu':wave.lib=corrected.lib;wave.dirs=corrected.dirs
    else:wave.module=corrected.module
    def launch(name,count,*args):
        if name=='matrix_bed_faces':args=(*args,p.eh.hiding,p.eh.d90)
        return old(name,count,*args)
    wave.launch=launch
    p.identity['river_mpm']=dict(source_sha256=digest(__file__),runtime=engine.identity,wave_runtime=corrected.identity,
        method='FLOW tranb2 + '+hiding,hiding_mode=2 if hiding!='none' else 1,
        resolved_auxiliaries='mixture D90, critical-Shields hiding once, FLOW spiral, KF slope, donor surface fraction once',
        wave='mean-state MPM radial derivative, frozen roughness/hiding coefficients; not full Hirano Jacobian',
        public_support=False)
    p.identity['river_mixtures']['capacity']=[c.identity for c in a.capacity]
    p.identity['river_mixtures']['hiding']=hiding
    p.identity['scope']='fixed_grid_classwise_MPM_layers_development'
    a.identity['selected_river_bundle']['scope']='fixed_grid_classwise_MPM_layers_implicit_midpoint'
    return a
