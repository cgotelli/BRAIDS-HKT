"""Locate the scalar limiter responsible for common-input amplification."""
import numpy as np
from stage_secondary185 import OUT,save


def main():
    f=np.load(OUT/'replay/sensitivity_01/RECONSTRUCTION.npz');rows=[]
    g={k[2:]:f[k] for k in f if k.startswith('g_')}
    for stage in (76,104,136,145):
        diff=abs(f[f'{stage}_cpu_grad']-f[f'{stage}_gpu_grad']);m,axis,c=np.unravel_index(np.argmax(diff),diff.shape)
        row=dict(stage=stage,primitive=int(m),cell=int(c),xy=g['xy'][:,c].tolist(),sides={})
        for backend in ('cpu','gpu'):
            v=f[f'{stage}_{backend}_primitive'];b=np.zeros(2)
            for k in range(g['offset'][c],g['offset'][c+1]):
                face=g['faces'][k];other=g['owner'][face] if g['signs'][k]<0 else g['neighbour'][face]
                if other>=0:b+=g['hydraulic_ls_weight'][k]*g['bed_ls_delta'][:,k]*(v[m,other]-v[m,c])
            a11,a12,a22,det=g['hydraulic_ls_matrix'][:,c]
            raw=np.array([a22*b[0]-a12*b[1],a11*b[1]-a12*b[0]])/det
            low=f[f'{stage}_{backend}_vlo'][m,c]-v[m,c];high=f[f'{stage}_{backend}_vhi'][m,c]-v[m,c]
            faces=[]
            for k in range(g['offset'][c],g['offset'][c+1]):
                face=g['faces'][k];d=g['mid'][:,face]-g['xy'][:,c];inc=raw@d
                phi=(high if inc>0 else low)/inc if inc else 1.
                faces.append(dict(face=int(face),displacement=d.tolist(),increment=float(inc),factor=float(phi)))
            row['sides'][backend]=dict(centre=float(v[m,c]),bounds=[float(low),float(high)],raw=raw.tolist(),
                limited=f[f'{stage}_{backend}_grad'][m,:,c].tolist(),faces=faces)
        rows.append(row)
    save(OUT/'replay/RECONSTRUCTION_DIAGNOSIS.json',rows)
    import json;print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
