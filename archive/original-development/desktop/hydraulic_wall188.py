"""Cross replay of saved hydraulic stages, including exterior bed history."""
import ctypes,json
import numpy as np


def run(a,path,folder):
    p=a.physics;xp=a.xp;f=np.load(path);out={}
    p.add_momentum=lambda *unused:None
    z=f['bed_00_z0'];exterior=f['bed_00_exterior_bed']
    a.z[:]=xp.asarray(z);p.z0[:]=a.z;p.exterior_bed[:]=xp.asarray(exterior)
    if a.backend=='cpu':
        fn=a.hydraulics.lib.bend_sediment_traces
        fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p];fn.restype=None
    for k in (0,20,26,31,43,71):
        q=xp.asarray(f[f'hydraulic_{k:02}_q']);a.q[:]=q
        a.midpoint.minimum_theta=1.;a.midpoint.frequency_peak=0.;a.midpoint.interval_base=q
        rate,water=a.midpoint.rate(q,.00001,200.2)
        out[f'{k}_rate']=a.host(rate);out[f'{k}_water']=a.host(water)
        out[f'{k}_exterior_flux']=a.host(a.exterior_bed.flux).copy()
        if a.backend=='gpu':
            b=a.hydraulics.reconstruction.reconstruct(q)
            for key,val in b.items():out[f'{k}_rec_{key}']=a.host(val).copy()
        else:
            trace=np.zeros((2,4,p.nf));fn(q.ctypes.data,a.z.ctypes.data,p.flow.threads,trace.ctypes.data)
            out[f'{k}_rec_face_value']=trace.transpose(1,0,2)
    np.savez(folder/'HYDRAULIC.npz',**out)
    (folder/'HYDRAULIC.json').write_text(json.dumps(dict(input=str(path),backend=a.backend,
        time=200.2,secondary_and_viscosity='excluded from rate for this operator diagnostic',
        exterior_state='saved accepted start of second sediment step'),indent=2))
    print('Hydraulic cross replay complete',folder,flush=True)


def sensitivity(a,path,folder):
    """Replay both trajectories on one backend; distinguish input from arithmetic."""
    p=a.physics;xp=a.xp;out={};rows=[]
    p.add_momentum=lambda *unused:None
    sources=[path,path.parent.parent/'onset_gpu/FIELDS.npz']
    for label,source in zip(('cpu','gpu'),sources):
        f=np.load(source);z=f['bed_00_z0']
        a.z[:]=xp.asarray(z);p.z0[:]=a.z;p.exterior_bed[:]=xp.asarray(f['bed_00_exterior_bed'])
        for k in (0,20,26,31,43,71):
            q=xp.asarray(f[f'hydraulic_{k:02}_q']);a.q[:]=q
            a.midpoint.minimum_theta=1.;a.midpoint.frequency_peak=0.;a.midpoint.interval_base=q
            rate,water=a.midpoint.rate(q,.00001,200.2)
            out[f'{label}_{k}_rate']=a.host(rate).copy()
            out[f'{label}_{k}_q']=a.host(q).copy();out[f'{label}_{k}_z']=z.copy()
            out[f'{label}_{k}_exterior_flux']=a.host(a.exterior_bed.flux).copy()
            b=a.hydraulics.reconstruction.reconstruct(q)
            for key,val in b.items():out[f'{label}_{k}_rec_{key}']=a.host(val).copy()
    for k,v in p.g.items():out['g_'+k]=a.host(v).copy()
    rec=a.hydraulics.reconstruction
    for k,v in rec.geometry.items():
        if isinstance(v,xp.ndarray):out['r_'+k]=a.host(v).copy()
    np.savez(folder/'SENSITIVITY.npz',**out)
    (folder/'RECONSTRUCTION.cu').write_text(rec.module.code)
