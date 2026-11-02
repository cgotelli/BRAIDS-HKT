"""Terrain-to-finest-field adapter, without importing QGIS in the worker."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from . import adaptive_river_worker as worker
from braids_gpu.atomic_output import write_json_atomic
from braids_core.windows_paths import filesystem_path

def prepare_recipe(recipe,output,stop=None,*,progress_callback=None,on_hierarchy=None,on_fields=None):
    if set(recipe)!={'profile','dem_path','initial'}:raise ValueError('Expected profile, dem_path and initial fields')
    p=recipe['profile'];c,m=worker.validate_profile(p);output=filesystem_path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    def progress(phase,percent):
        worker.check_stop(stop);write_json_atomic(output/'PROGRESS.json',dict(phase=phase,percent=percent,status='preparing'))
        if progress_callback is not None:progress_callback(phase,percent)
    try:
        progress('hierarchy',2);tree=worker.hierarchy(p,stop)
        required=()
        if on_hierarchy is not None:
            progress('bank survey mapping',12);required=on_hierarchy(tree) or ()
        progress('terrain',15)
        path=worker.ROOT/'braids_qgis/core/raster_surface.py'
        spec=importlib.util.spec_from_file_location('_braids_river_surface',path);surface=importlib.util.module_from_spec(spec);spec.loader.exec_module(surface)
        z,audit=surface.surface_averages(tree.finest,recipe['dem_path'],progress=lambda v:progress('terrain',15+.35*v),
            cancelled=lambda:stop is not None and filesystem_path(stop).exists(),threads=int(p['controls']['threads']))
        initial=recipe['initial'];expected={'stage_m','velocity_m_s','layer_thickness_m','layer_fractions','concentration_kg_m3'}
        if set(initial)!=expected:raise ValueError('Initial field recipe differs')
        ng=len(m.fraction_ids);n=len(z);th=np.asarray(initial['layer_thickness_m'],float);fr=np.asarray(initial['layer_fractions'],float)
        velocity=np.asarray(initial['velocity_m_s'],float);conc=np.asarray(initial['concentration_kg_m3'],float)
        if velocity.shape!=(2,) or conc.shape!=(ng,) or fr.shape!=(len(th),ng):raise ValueError('Initial grain/layer/velocity shape differs')
        h=np.maximum(float(initial['stage_m'])-z,0.);q=np.vstack((h,velocity[:,None]*h));q[1:,h<=1e-6]=0
        concentration=np.broadcast_to(conc[:,None],(ng,n)).copy();concentration[:,h<=1e-6]=0
        fields=dict(q=q,rock=z-th.sum(),thickness=np.broadcast_to(th[:,None],(len(th),n)).copy(),
            fractions=np.broadcast_to(fr[:,:,None],(len(th),ng,n)).copy(),concentration=concentration)
        worker.arrays_valid(fields,tree,m);progress('native preparation',55)
        if on_fields is not None:on_fields(tree,fields)
        result=worker.prepare(p,fields,output/'prepared',stop_file=stop,required_leaves=required)
        write_json_atomic(output/'TERRAIN.json',audit);write_json_atomic(output/'RECIPE.json',recipe)
        progress('prepared',100);return result
    except BaseException as exc:
        write_json_atomic(output/'STATUS.json',dict(status='cancelled' if isinstance(exc,InterruptedError) else 'failed',error=str(exc)))
        raise
