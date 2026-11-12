"""Read-only execution-order traces plus explicit conservative input perturbation.

This is a saved-state diagnostic, not a supported restart or physics option.
"""
import json
import numpy as np


class WallAudit:
    def __init__(self,a,perturb=0.):
        self.a=a;self.rows=[];self.times=[];self.labels=[];p=a.physics;g=p.g
        xy=a.host(g['xy']);o=a.host(g['owner']);j=a.host(g['neighbour'])
        seeds=[3570,int(np.argmin((xy[0]-11.903529)**2+(xy[1]-2.051163)**2))]
        faces=np.flatnonzero(np.isin(o,seeds)|np.isin(j,seeds))
        cells=np.unique(np.r_[o[faces],j[faces]]);cells=cells[cells>=0]
        self.cells=cells;self.faces=faces;self.geometry={k:a.host(v).copy() for k,v in g.items()}
        self.perturbation=dict(depth_m=perturb)
        if perturb:
            c=seeds[0];adj=faces[(j[faces]>=0)&((o[faces]==c)|(j[faces]==c))]
            f=int(adj[0]);other=int(j[f] if o[f]==c else o[f]);area=a.host(g['area'])
            for cell,dh in ((c,perturb),(other,-perturb*area[c]/area[other])):
                h=float(a.q[0,cell]);a.q[:,cell]*=(h+dh)/h;p.W[cell]*=(h+dh)/h
            self.perturbation.update(cells=[c,other],water_volume_delta=0.,velocity_and_intensity_preserved=True)
        # Class loop owns the full cell -> reconstruction -> wave -> slope route.
        old=a.bed_faces
        def bed(*args,**kw):
            self.capture('bed_input',{'q':args[0],'z':args[1],'hi':args[2]})
            out=old(*args,**kw)
            self.capture('bed_output',{k:getattr(p,k) for k in (
                'bedcell','capacity_gradient','capacity_limiter','bedface','bedcorrected',
                'effective_fields','effective_q','wave_diagnostic','wave_physical_normal',
                'class_flux','hydraulic_wave_trace','hydraulic_face_opening')})
            return out
        a.bed_faces=bed
        old_rec=p.reconstruction.launch
        def reconstructed(name,count,*args):
            out=old_rec(name,count,*args)
            self.capture(name,{k:getattr(p,k) for k in ('capacity_gradient','capacity_limiter','bedface')})
            return out
        p.reconstruction.launch=reconstructed
        old_wave=p.wave_engine.launch
        def wave(name,count,*args):
            out=old_wave(name,count,*args)
            if name=='matrix_bed_faces':
                self.capture(name,{k:getattr(p,k) for k in ('wave_state_gradient','bedface','wave_diagnostic')})
            return out
        p.wave_engine.launch=wave
        old_apply=a._class_update.apply
        def apply(*args,**kw):
            out=old_apply(*args,**kw)
            self.capture('exner',dict(class_delta=p.class_delta,class_flux=p.class_flux,factors=p.factors))
            return out
        a._class_update.apply=apply

    def capture(self,label,arrays):
        p=self.a.physics;row={}
        for k,v in arrays.items():
            idx=self.cells if v.shape[-1]==p.n else self.faces if v.shape[-1]==p.nf else None
            if idx is not None:row[k]=self.a.host(v[...,idx]).copy()
        self.rows.append(row);self.times.append(p.time);self.labels.append(label)

    def save(self,folder):
        fields=dict(cells=self.cells,faces=self.faces,times=self.times,labels=self.labels)
        for k,v in self.geometry.items():fields['g_'+k]=v
        for i,row in enumerate(self.rows):
            for k,v in row.items():fields[f'{i:05}_{k}']=v
        np.savez(folder/'WALL_TRACE.npz',**fields)
        (folder/'PERTURBATION.json').write_text(json.dumps(self.perturbation,indent=2),encoding='utf-8')
