"""Independent mathematical checks of the isolated gradient projection."""
import ctypes,json,sys,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'validation/_inlet187/prescribed_candidate'),str(ROOT)]
from projected_gradient188 import HELPER


def reference(g,A,b):
    candidates=[np.zeros(2),g]
    for i,n in enumerate(A):
        candidates.append(g-(n@g-b[i])*n)
        for j in range(i):
            m=np.array([n,A[j]])
            if abs(np.linalg.det(m))>1.e-10:candidates.append(np.linalg.solve(m,b[[i,j]]))
    feasible=[v for v in candidates if np.all(A@v<=b+2.1e-14)]
    return min(feasible,key=lambda v:np.sum((v-g)**2))


def main():
    import cupy as cp
    from validation.gpu_fwave.reconstruction import OPTIONS
    candidate='--candidate' in sys.argv
    helper=HELPER;symbol='__cutcell_solver_MOD_project_pair188'
    if candidate:
        runtime=ROOT/'validation/_wall188/candidate_v3'
        table=json.loads((runtime/'PRECOMPILED.json').read_text())
        row=next(v for k,v in table.items() if 'shoreline_hydraulic_probe.build:' in k)
        library=runtime/row['native']['file'];dll_dir=os.add_dll_directory(str(library.parent))
        helper=(runtime/'braids_core/sediment/directional_hydraulic_limiter.cu.inc').read_text().split('// CELL_INSERT\n')[0]
        helper=helper.replace('braids_project_pair','project_gradient')
        symbol='__cutcell_solver_MOD_braids_project_pair'
    else:
        folder=ROOT/'validation/_wall188/replay/projected_cpu4'
        receipt=json.loads((folder/'PROJECTED.json').read_text())
        dll_dir=os.add_dll_directory(str(Path(receipt['command'][0]).parent))
        library=receipt['command'][receipt['command'].index('-o')+1]
    lib=ctypes.CDLL(str(library))
    fn=getattr(lib,symbol);fn.argtypes=[ctypes.c_void_p]*6;fn.restype=None
    code=helper+'''extern "C" __global__ void batch(int n,const double* in,const double* ax,const double* ay,const double* b,const int* count,double* out){
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
    double x=in[2*c],y=in[2*c+1];project_gradient(x,y,count[c],ax+64*c,ay+64*c,b+64*c);
    out[2*c]=x;out[2*c+1]=y;}'''
    mod=cp.RawModule(code=code,options=OPTIONS,name_expressions=('batch',))
    rng=np.random.default_rng(188);cases=[]
    for i in range(160):
        angle=rng.uniform(0,2*np.pi,6);A=np.c_[np.cos(angle),np.sin(angle)]
        b=rng.uniform(0.,2.,6);g=rng.normal(size=2)*3
        cases.append((g,A,b))
        turn=.713;R=np.array([[np.cos(turn),-np.sin(turn)],[np.sin(turn),np.cos(turn)]])
        cases.append((R@g,A@R.T,b))
    # Exact constant and an unrestricted affine slope must be untouched.
    cases.extend([(np.zeros(2),np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]),np.zeros(4)),
                  (np.array([.03,.02]),np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]),np.ones(4))])
    n=len(cases);input=np.array([g for g,A,b in cases]);ax=np.zeros((n,64));ay=ax.copy();bounds=ax.copy();count=np.zeros(n,np.int32)
    host=[];exact=[]
    for c,(g,A,b) in enumerate(cases):
        count[c]=len(b);ax[c,:len(b)]=A[:,0];ay[c,:len(b)]=A[:,1];bounds[c,:len(b)]=b
        x=np.array([g[0]]);y=np.array([g[1]]);k=np.array([len(b)],np.int32)
        fn(x.ctypes.data,y.ctypes.data,k.ctypes.data,ax[c].ctypes.data,ay[c].ctypes.data,bounds[c].ctypes.data)
        host.append([x[0],y[0]]);exact.append(reference(g,A,b))
    device=[cp.asarray(v) for v in (input,ax,ay,bounds,count)];out=cp.empty_like(device[0])
    mod.get_function('batch')(((n+127)//128,),(128,),(np.int32(n),*device,out))
    host=np.array(host);gpu=cp.asnumpy(out);exact=np.array(exact)
    residual=max(float(np.max(A@v-b)) for v,(g,A,b) in zip(host,cases))
    rotation=max(float(np.max(abs(R@host[i]-host[i+1]))) for i in range(0,320,2))
    result=dict(cases=n,max_cpu_gpu=float(np.max(abs(host-gpu))),max_independent_reference=float(np.max(abs(host-exact))),
        max_constraint_excess=residual,max_rotation_difference=rotation,constant_exact=bool(np.array_equal(host[-2],[0.,0.])),
        affine_exact=bool(np.array_equal(host[-1],input[-1])))
    assert result['max_cpu_gpu']<1e-12 and result['max_independent_reference']<1e-12
    assert residual<3e-14 and rotation<1e-12 and result['constant_exact'] and result['affine_exact']
    (ROOT/('validation/_wall188/PACKAGED_UNIT.json' if candidate else 'validation/_wall188/PROJECTED_UNIT.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
