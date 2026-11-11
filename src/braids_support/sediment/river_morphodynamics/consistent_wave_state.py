"""Research adapter: sediment-wave hydraulics use the hydraulic face producer.

Only the hydraulic traces in the dissipative bed-wave term change. Physical
sediment MUSCL reconstruction, formulas, slope transformation and Exner stay
unchanged. Traces are recomputed from the sediment call's exact q,z, never taken
from a previous Euler stage. Exterior sediment boundary fields are untouched.
"""
import ctypes
from pathlib import Path
import numpy as np
from braids_core.sediment.bend_methods_v1 import digest
from braids_core.sediment import bend_native_hydraulics as native
from .sediment_wave import WaveEngine


def source(pressure_only=False):
    code=Path(__file__).with_name('matrix_wave.h').read_text()
    if pressure_only:
        code=code.replace('double*diagnostic){','double*diagnostic,const double*hydro){')
        needle='s[1][0]*(s[1][1]*s[1][1]+.5*9.81*s[1][0])-s[0][0]*(s[0][1]*s[0][1]+.5*9.81*s[0][0])+gh*(s[1][3]-s[0][3])'
        assert code.count(needle)==1
        return code.replace(needle,'s[1][0]*s[1][1]*s[1][1]-s[0][0]*s[0][1]*s[0][1]+gh*(hydro[4*nf+i]-hydro[i])')
    needle='   for(int c=0;c<4;++c)s[side][c]=value(c,a,n,q,z)+stategrad[2*c*n+a]*dx+stategrad[(2*c+1)*n+a]*dy;'
    assert code.count(needle)==1
    return code.replace(needle,'''   // stategrad is the fresh hydraulic primitive face array (side,eta,u,v,z).
   s[side][0]=fmax(1.e-6,stategrad[(4*side)*nf+i]-stategrad[(4*side+3)*nf+i]);
   s[side][1]=stategrad[(4*side+1)*nf+i];s[side][2]=stategrad[(4*side+2)*nf+i];
   s[side][3]=stategrad[(4*side+3)*nf+i];''')


def install(a,folder,*,pressure_only=False):
    p=a.physics;xp=a.xp
    if p.time or p.steps or p._open:raise ValueError('Install before execution')
    if hasattr(p,'hydraulic_wave_trace'):raise ValueError('Fresh trace adapter already installed')
    from braids_core.windows_paths import filesystem_path
    target=filesystem_path(folder)/('consistent_wave_pressure.h' if pressure_only else 'consistent_wave_state.h');target.parent.mkdir(parents=True,exist_ok=True);code=source(pressure_only)
    if target.exists():assert target.read_text()==code
    else:target.write_text(code)
    engine=WaveEngine(a.backend,p.flow.threads,matrix=True,source=target)
    # Explicit kernel-ABI capability. Archived launch closures have no cache
    # arguments and must never be rebound to the longer cached entry point.
    engine._class_wave_cache_dispatch=True
    original=engine.launch;p.hydraulic_wave_trace=xp.zeros((2,4,p.nf));p.hydraulic_wave_calls=0
    if a.backend=='cpu':
        fn=a.hydraulics.lib.bend_sediment_traces
        fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p];fn.restype=None
        def update():
            with native._lock:
                fn(p.wave_q.ctypes.data,p.z0.ctypes.data,p.flow.threads,p.hydraulic_wave_trace.ctypes.data)
    else:
        rec=a.hydraulics.reconstruction
        def update():
            rec.geometry['bed'][...]=p.z0
            b=rec.reconstruct(p.wave_q)
            if bool(xp.any(b['primitive_status'])) or bool(xp.any(b['reconstruction_status'])):
                raise ValueError('Invalid fresh hydraulic sediment-wave trace')
            p.hydraulic_wave_trace[:]=b['face_value'].transpose(1,0,2)
    def launch(name,count,*args):
        if name=='wave_state_gradients':
            return original(name,count,*args) if pressure_only else None
        if name!='matrix_bed_faces':raise ValueError(name)
        batch=getattr(p,'_river_hydraulic_batch',None)
        if batch is None or not batch['traces']:
            update();p.hydraulic_wave_calls+=1
            continuation=getattr(p,'open_bed_trace',None)
            if continuation is not None and not pressure_only:
                continuation.apply(p.z0,p.grad,p.hydraulic_wave_trace)
            if batch is not None:batch['traces']=True
        values=list(args)
        if pressure_only:values.append(p.hydraulic_wave_trace)
        else:values[7]=p.hydraulic_wave_trace
        if hasattr(engine,'_class_shared'):
            # Grain classes share q,z and effective roughness only inside the
            # guarded batch. Never reuse across calls, retries or mesh owners.
            reuse=batch is not None and batch.get('wave_mean',False)
            values.extend((int(reuse),engine._class_shared))
            result=original(name,count,*values)
            if batch is not None:batch['wave_mean']=True
            return result
        return original(name,count,*values)
    engine.launch=launch;p.wave_engine=engine
    p.identity['consistent_hydraulic_wave_state']=dict(wrapper_sha256=digest(__file__),source_sha256=digest(target),
        runtime=engine.identity,state='fresh hydraulic eta/u/v/z face values at current sediment q,z',
        pressure_only=pressure_only,
        physical_capacity_reconstruction='unchanged',hydraulic_core_changed=False,public_support=False)
    return a
