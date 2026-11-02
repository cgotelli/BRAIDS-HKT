"""Shared CPU/GPU cell-local hydraulic drag; face sediment conversion unchanged."""
from pathlib import Path
from types import MethodType
from braids_core.sediment import river_roughness
from braids_core.sediment.bend_methods_v1 import digest
from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine

METHOD='cell_local_hydraulic_resistance_v1'


def source(spatial=False):
    code=Path(__file__).with_suffix('.h').read_text()
    if spatial:
        code=code.replace(' double value,int feedback,',' const double*values,int feedback,').replace('LOOP(n){double h=q[i];','LOOP(n){double value=values[i],h=q[i];')
    return river_roughness.source(spatial)+'\n'+code


def install(owner,folder):
    p=owner.physics
    if owner.time or owner.steps or p._open:raise ValueError('Fresh owner required')
    selected=p.identity['river_roughness'];feedback=p.identity['river_bedforms']['enabled']
    mode,value=river_roughness.selected(selected['method'],selected['parameters'])
    target=Path(folder)/'hydraulic_drag.h';target.parent.mkdir(parents=True,exist_ok=True)
    spatial=hasattr(p,'spatial_roughness')
    if spatial:value=p.spatial_roughness
    code=source(spatial)
    if target.exists():
        if target.read_text()!=code:raise ValueError('Hydraulic drag source differs')
    else:target.write_text(code)
    engine=WaveEngine(owner.backend,p.flow.threads,source=target,names=('river_cell_drag',))
    p.hydraulic_chezy=owner.xp.empty(p.n);previous=p.friction
    def friction(self,q,dt,*args):
        # Preserve the existing trial-owned equilibrium predictor. It changes
        # state only at accepted-start prediction, not inside an open trial.
        if feedback:previous(q,0.,*args)
        engine.launch('river_cell_drag',p.n,p.n,float(dt),p.parameters.gravity,p.parameters.dry,
            mode,value,int(feedback),p.bedform_state,q,p.hydraulic_chezy)
    p.friction=MethodType(friction,p)
    p.identity['hydraulic_resistance']=dict(method=METHOD,runtime=engine.identity,
        wrapper_sha256=digest(__file__),geometry='cell-average physical depth; no adjacent-face wetness dependency',
        constitutive_owner=selected['total_resistance_owner'],
        sediment_mapping='unchanged face-based shear/roughness conversion',explicit_backend=owner.backend)
    # Initialize a derived diagnostic, without advancing bedform history or
    # zeroing momentum in the caller's initial state.
    engine.launch('river_cell_drag',p.n,p.n,0.,p.parameters.gravity,p.parameters.dry,
        mode,value,int(feedback),p.bedform_state,owner.q.copy(),p.hydraulic_chezy)
    return owner
