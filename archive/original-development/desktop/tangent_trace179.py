"""Causal trace substitution: release only wall-tangential velocity gradients.

This is an operator diagnostic, not an installed or coupled numerical option.
"""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
import trace179 as trace
from check_jet173 import setup,document
from quadrature176 import Flux,gather
from tip178 import exact_rate
from stage_wall172 import save


def main():
    from braids_core import river_worker
    setup();out=trace.BASE/'tangent_diagnostic';out.mkdir(exist_ok=False)
    a=dict(np.load(ROOT/'validation/_tip178/probes/gpu/bounded.npz'))
    original_linear=trace.linear
    def linear(a):
        v,grad,walls=original_linear(a)
        for c,fs in enumerate(walls):
            if not fs:continue
            n=a['normal'][:,fs[0]]
            if any(abs(np.dot(n,a['normal'][:,f]))<1-1e-12 for f in fs):continue
            t=np.array([-n[1],n[0]]);old=a['rec_grad'][1:3,:,c]
            grad[1:3,:,c]=old+np.outer(t,t)@(grad[1:3,:,c]-old)
        return v,grad,walls
    trace.linear=linear
    original=river_worker.core.factory
    def factory(*args,**kw):
        make=original(*args,**kw)
        def build(*args,**kw):
            o=make(*args,**kw);values=trace.faceclip(a,o.data)
            l,r=Flux(o)(a,values);rate=gather(a,l,r)
            ids=[np.flatnonzero(np.all(a['xy']==p,axis=1))[0] for p in trace.POINTS]
            truth=np.array([exact_rate(*(np.array(p)-[8,3])) for p in trace.POINTS]).T;e=rate[:,ids]-truth
            result=dict(mass_rms=float(np.sqrt(np.mean(e[0]**2))),momentum_rms=float(np.sqrt(np.mean(e[1:]**2))),
                rates=rate[:,ids].T.tolist(),coupled_qualified=False)
            np.savez(out/'OPERATOR.npz',trace=values,rate=rate);save(out/'CHECK.json',result);print(result,flush=True)
            return o
        build.__dict__.update(make.__dict__);return build
    river_worker.core.factory=factory;save(out/'project.json',document('cpu'))
    p=river_worker.prepare(out/'project.json',out/'prepared');river_worker.run(p,out/'construction',until=.001)


if __name__=='__main__':main()
