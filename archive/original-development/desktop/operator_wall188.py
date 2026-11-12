"""Frozen-state continuity audit through hydraulic and bedload operators."""
import json
import numpy as np


def run(a,base,peer,folder):
    p=a.physics;xp=a.xp;g=p.g;data={};rows=[]
    initial=a.snapshot()
    def evaluate(q,z,hi):
        a.restore(initial);a.q[:]=xp.asarray(q);a.z[:]=xp.asarray(z);p.W[:]=xp.asarray(hi)
        p.friction(a.q,0.);p.begin(a.q,a.z,.2,a.time)
        a.midpoint.minimum_theta=1.;a.midpoint.frequency_peak=0.;a.midpoint.interval_base=a.q
        rate,water=a.midpoint.rate(a.q,.0001,a.time)
        before=a.host(rate).copy()
        a.bed_faces(a.q,a.z,p.W,water)
        fields={k:a.host(getattr(p,k)).copy() for k in ('bedcell','capacity_gradient',
            'capacity_limiter','bedface','bedcorrected','class_flux','effective_fields',
            'effective_q','hydraulic_wave_trace','wave_diagnostic','inputs','outputs')}
        fields['hydraulic_rate']=before
        flux=fields['class_flux'][0];length=a.host(g['length']);area=a.host(g['area'])
        own=a.host(g['owner']);nb=a.host(g['neighbour'])
        div=-np.bincount(own,weights=flux*length,minlength=p.n)
        inside=nb>=0;div+=np.bincount(nb[inside],weights=flux[inside]*length[inside],minlength=p.n)
        fields['bed_rate']=div/area*p.morfac/(1.-p.parameters.porosity)
        return fields
    first=evaluate(base['q'],base['z'],base['W'])
    for k,v in first.items():data['base_'+k]=v
    for group in ('q','z','W','all'):
        for scale in (1e-6,1e-4,.01,1.):
            inputs={k:base[k]+(scale*(peer[k]-base[k]) if group in (k,'all') else 0.) for k in ('q','z','W')}
            result=evaluate(inputs['q'],inputs['z'],inputs['W'])
            maxima={k:float(np.max(np.abs(v-first[k]))) for k,v in result.items()}
            peak={k:np.unravel_index(np.argmax(np.abs(v-first[k])),v.shape)[-1].item() for k,v in result.items()}
            rows.append(dict(group=group,scale=scale,maxima=maxima,peak=peak))
            for k,v in result.items():data[f'{group}_{scale:g}_'+k]=v
    for k,v in g.items():data['g_'+k]=a.host(v)
    np.savez(folder/'OPERATOR.npz',**data)
    (folder/'OPERATOR.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print('Operator ladder complete',folder,flush=True)
