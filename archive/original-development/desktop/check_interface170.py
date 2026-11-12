"""Read-only augmented-wave audit, with actual CPU/CUDA analytical controls.

The middle augmented wave is a normal-momentum correction, NOT a contact.
No production source, saved state, or Delft3D file is changed.
"""
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from check_bedload161 import save,sha

LIB=ROOT/'desktop/runtime/precompiled/d440a06059c179ec44ed/hydraulics.dll'
CUDA=ROOT/'desktop/runtime/validation/gpu_fwave/augmented_fwave.cu'
G=9.81


def cpu_solver(states):
    """Call the exact exported rpn2 in the delivered native hydraulic DLL."""
    lib=ctypes.CDLL(str(LIB))
    ctypes.c_double.in_dll(lib,'__geoclaw_module_MOD_grav').value=G
    ctypes.c_double.in_dll(lib,'__geoclaw_module_MOD_dry_tolerance').value=1e-6
    fun=lib.rpn2_;fun.restype=None;fun.argtypes=[ctypes.c_void_p]*15
    dims=[ctypes.c_int(v) for v in (1,1,3,3,1,1,0)]
    result=[]
    for row in states.T:
        ql=np.asfortranarray(np.tile(row[4:7,None],(1,3)))
        qr=np.asfortranarray(np.tile(row[:3,None],(1,3)))
        al=np.full((1,3),row[7],order='F');ar=np.full((1,3),row[3],order='F')
        fw=np.zeros((3,3,3),order='F');sw=np.zeros((3,3),order='F')
        am=np.zeros((3,3),order='F');ap=np.zeros((3,3),order='F')
        fun(*[ctypes.byref(x) for x in dims],*[a.ctypes.data for a in (ql,qr,al,ar,fw,sw,am,ap)])
        result.append(np.r_[am[:,1],ap[:,1],sw[:,1],fw[:,:,1].T.reshape(-1)])
    return np.array(result).T


def gpu_solver(states):
    import cupy as cp
    from validation.gpu_fwave.reconstruction import OPTIONS
    module=cp.RawModule(code=CUDA.read_text(),options=OPTIONS)
    fn=module.get_function('augmented_fwave_local');n=states.shape[1]
    out=cp.zeros((18,n));status=cp.zeros(n,cp.int32)
    fn(((n+255)//256,),(256,),(cp.asarray(states),np.int64(n),np.float64(G),np.float64(1e-6),out,status))
    cp.cuda.Stream.null.synchronize()
    assert not cp.asnumpy(status).any()
    return cp.asnumpy(out)


def physical(local):
    h,mn,mt,_=local
    return np.array([mn,mn*mn/h+.5*G*h*h,mn*mt/h])


def split(local,output):
    """Exact local-coordinate decomposition, not a replacement Riemann solver.

    Gravity waves retain tangential momentum carried with their mass change.
    The residual tangential flux jump is the advected contact contribution;
    stock GeoClaw places it in the outer wave selected by intermediate mass flux.
    """
    speed=output[6:9];fw=output[9:].reshape(3,3).copy()
    vl=local[2]/local[0];vr=local[6]/local[4]
    groups=np.zeros((4,3,3)) # acoustic left, augmented middle, acoustic right, contact
    groups[0,0]=fw[0];groups[0,0,2]=fw[0,0]*vl
    groups[1,1]=fw[1]
    groups[2,2]=fw[2];groups[2,2,2]=fw[2,0]*vr
    groups[3]=fw-groups[:3].sum(0)
    residual=local[5]*vr-local[1]*vl-fw[0,0]*vl-fw[2,0]*vr
    branch=0 if local[1]+fw[0,0]<=0 else 2
    expected=np.zeros((3,3));expected[branch,2]=residual
    np.testing.assert_allclose(groups[3],expected,atol=2e-14,rtol=2e-13)
    left=np.where(speed < -1e-14,1.,np.where(speed > 1e-14,0.,.5))
    right=1-left
    am=(groups*left[None,:,None]).sum(1)
    ap=(groups*right[None,:,None]).sum(1)
    np.testing.assert_allclose(am.sum(0),output[:3],atol=2e-14,rtol=2e-13)
    np.testing.assert_allclose(ap.sum(0),output[3:6],atol=2e-14,rtol=2e-13)
    fl=physical(local[:4]);fr=physical(local[4:])
    # Exact left/right interface flux: central physical + signed waves +
    # half the source imbalance. This term matters over a bed discontinuity.
    source=output[:3]+output[3:6]-(fr-fl)
    correction=.5*(am-ap)
    return dict(groups=groups,correction=correction,source=source,
                central=.5*(fl+fr),branch=branch,residual=residual,speed=speed)


def global_vector(v,n):
    return np.stack([v[...,0],v[...,1]*n[0]-v[...,2]*n[1],
                     v[...,1]*n[1]+v[...,2]*n[0]],axis=-1)


def cell_ledger(a,c):
    """Exact face ledger and velocity acceleration, retaining source and wall projection."""
    faces=np.flatnonzero((a['owner']==c)|(a['neighbour']==c));rates=np.zeros((8,3));rows=[]
    for f in faces:
        side=0 if a['owner'][f]==c else 1;sign=1-2*side;n=a['normal'][:,f]
        assert not a['closed'][f]
        assert a['neighbour'][f]>=0 or a['bid'][f]<0
        d=split(a['local'][:,f],a['riemann'][:,f]);central=d['central'].copy()
        hl=a['local'][0,f];hr=a['local'][4,f]
        pressure=np.array([0.,.25*G*(hl*hl+hr*hr),0.])
        parts=np.vstack([central-pressure,pressure,d['correction'],.5*sign*d['source'],np.zeros(3)])
        parts=global_vector(parts,n)
        if a['neighbour'][f]<0:
            # Apply exactly the linear no-water/no-tangent wall projection to all terms.
            parts[:,0]=0;parts[:,1:]=np.outer(parts[:,1:]@n,n)
        trace=a['rec_trace'];h=a['q'][0,c]
        assert not a['rec_hydro_constant'][c]
        corr=.5*G*(h+trace[8+side,f])*(trace[3+4*side,f]-a['z'][c])
        parts[-1]=np.r_[0.,corr*n]
        actual=a['fl' if side==0 else 'fr'][:,f]
        np.testing.assert_allclose(parts.sum(0),actual,atol=2e-13,rtol=2e-13)
        rate=-sign*a['length'][f]/a['area'][c]*parts;rates+=rate
        local=a['local'][:,f];mass=physical(local[:4])[0]+a['riemann'][0,f]
        expected_t=mass*(local[2]/local[0] if mass>=0 else local[6]/local[4])
        actual_t=physical(local[:4])[2]+a['riemann'][2,f]
        rows.append(dict(face=int(f),xy=a['mid'][:,f].tolist(),normal=n.tolist(),side=side,
            left=local[:4].tolist(),right=local[4:].tolist(),speeds=d['speed'].tolist(),
            contact_branch=d['branch'],contact_residual=float(d['residual']),
            rate_pieces=rate.tolist(),tangential_flux=float(actual_t),
            mass_upwind_tangential_flux=float(expected_t),
            transverse_upwind_error=float(abs(actual_t-expected_t))))
    np.testing.assert_allclose(rates.sum(0),a['rate'][:,c],atol=3e-12,rtol=2e-13)
    uv=a['q'][1:,c]/a['q'][0,c]
    acc=(rates[:,1:]-rates[:,:1]*uv)/a['q'][0,c]
    return dict(cell=int(c),xy=a['xy'][c].tolist(),rate_pieces=rates.tolist(),
                acceleration_pieces=acc.tolist(),faces=rows,
                reconstruction_error=float(abs(rates.sum(0)-a['rate'][:,c]).max()))


def controls():
    rows=[];states=[];expected=[]
    for h in (.1,.5,2.):
        for un in (-6.,-.6,0.,.6,6.):
            for vt in ((1.,3.),(3.,1.),(2.,2.)):
                states.append([h,h*un,h*vt[0],0.,h,h*un,h*vt[1],0.])
                expected.append([h*un,h*un*un+.5*G*h*h,h*un*(vt[0] if un>=0 else vt[1])])
                rows.append(dict(h=h,un=un,tangent=list(vt),kind='contact'))
    # Exact one-directional acoustic linear limit around rest.
    for direction in (-1,1):
        for eps in (1e-4,1e-6,1e-8):
            h=.5;c=np.sqrt(G*h)
            states.append([h,0.,0.,0.,h+eps,direction*c*eps,0.,0.])
            base=np.array([0.,.5*G*h*h,0.])
            expected.append(base if direction==1 else base+np.array([-c*eps,G*h*eps,0.]))
            rows.append(dict(h=h,epsilon=eps,direction=direction,kind='acoustic_linear_limit'))
    x=np.ascontiguousarray(np.array(states).T);cpu=cpu_solver(x);gpu=gpu_solver(x)
    np.testing.assert_allclose(cpu,gpu,rtol=5e-13,atol=2e-12)
    for i,row in enumerate(rows):
        fl=physical(x[:4,i])+cpu[:3,i];fr=physical(x[4:,i])-cpu[3:6,i]
        row['flux_max_error']=float(abs(fl-expected[i]).max())
        row['shared_flux_error']=float(abs(fl-fr).max())
        row['backend_max_error']=float(abs(cpu[:,i]-gpu[:,i]).max())
        row['expected']=np.asarray(expected[i]).tolist();row['actual']=fl.tolist()
        if row['kind']=='contact':assert row['flux_max_error']<2e-12
        else:assert row['flux_max_error']<20*row['epsilon']**2+3e-15
        d=split(x[:,i],gpu[:,i]);row['split_error']=float(abs(d['central']+d['correction'].sum(0)+.5*d['source']-fl).max())
        # Global coordinate rotation changes only vector representation; test
        # actual local input formation as in wave_flux, including normal sign.
    return x,cpu,gpu,rows


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='v1');args=ap.parse_args()
    out=ROOT/f'validation/_interface170/{args.tag}';out.mkdir(parents=True,exist_ok=False)
    x,cpu,gpu,checks=controls();np.savez(out/'CONTROLS.npz',states=x,cpu=cpu,gpu=gpu)
    cases={};inputs={}
    for name in ('BRAIDS','FLOW_PRIMARY'):
        p=ROOT/f'validation/_corner167/baseline/gpu/{name}.npz';inputs[str(p.relative_to(ROOT))]=sha(p)
        a=np.load(p);sel=np.flatnonzero((a['xy'][:,0]>7.5)&(a['xy'][:,0]<10.5)&(a['xy'][:,1]>3)&(a['xy'][:,1]<4.5))
        ledger=[cell_ledger(a,c) for c in sel]
        total=sum(np.array(c['rate_pieces'])*a['area'][c['cell']] for c in ledger)
        volume=float(np.sum(a['q'][0,sel]*a['area'][sel]));u=(a['q'][1:,sel]*a['area'][sel]).sum(1)/volume
        acc=(total[:,1:]-total[:,:1]*u)/volume
        cases[name]=dict(cells=ledger,region_acceleration=acc.tolist(),volume=volume)
        # Re-evaluate all saved wet interface states on the real CPU and GPU
        # solvers, independent of the stored GPU diagnostic buffers.
        fs=np.unique([f['face'] for c in ledger for f in c['faces']]);local=np.ascontiguousarray(a['local'][:,fs])
        cr=cpu_solver(local);gr=gpu_solver(local)
        np.testing.assert_allclose(gr,a['riemann'][:,fs],atol=2e-13,rtol=2e-13)
        np.testing.assert_allclose(cr,gr,atol=2e-13,rtol=2e-13)
        cases[name]['replay_max_error']=float(abs(gr-a['riemann'][:,fs]).max())
        cases[name]['replay_backend_max_error']=float(abs(cr-gr).max())
        np.savez(out/(name+'.npz'),faces=fs,local=local,cpu=cr,gpu=gr)
    save(out/'AUDIT.json',dict(controls=checks,cases=cases,inputs=inputs,
        components=['central_advection','central_pressure','left_acoustic','augmented_middle',
                    'right_acoustic','contact','source_imbalance','bed_quadrature'],
        sources={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),LIB,CUDA,ROOT/'desktop/runtime/RUNTIME.json')},
        production_changed=False,Delft3D_modified=False))
    print(json.dumps(dict(checks=len(checks),cases={k:v['region_acceleration'] for k,v in cases.items()}),indent=2))


if __name__=='__main__':main()
