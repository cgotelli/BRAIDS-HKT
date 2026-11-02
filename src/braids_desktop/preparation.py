"""Desktop preparation using verified geometry/DEM assets and stock core APIs.

Time stepping, native records, physics and all runnable receipts remain those of
the authenticated runtime. Cached assets contain JSON/numeric data, never code.
"""
from dataclasses import asdict
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import numpy as np
from braids_core import river_worker as public
from braids_core.sediment import adaptive_river_worker as worker
from braids_core.sediment.river_leaf_io import sha,publish_directory
from braids_gpu.atomic_output import write_json_atomic,save_arrays_atomic
from braids_core.river_project import identity,load_project,worker_profile


def _tuples(value):return tuple(_tuples(v) for v in value) if isinstance(value,list) else value


def encode_tree(tree):
    meshes=[]
    for mesh in tree.meshes:
        meshes.append(dict(domain_id=mesh.domain_id,crs_wkt=mesh.crs_wkt,grid=asdict(mesh.grid),
            cells=[asdict(c) for c in mesh.cells],faces=[asdict(f) for f in mesh.faces],
            obstacle_ids=mesh.obstacle_ids,solid_area_m2=mesh.solid_area_m2))
    return dict(meshes=meshes,offsets=tree.offsets,parent=tree.parent.tolist(),children=tree.children,
                node_level=tree.node_level.tolist(),maximum_level=tree.maximum_level.tolist(),refinement_ratio=tree.refinement_ratio)


def decode_tree(value):
    from braids_core.boundaries.amr import AMRHierarchy
    from braids_core.boundaries.cut_cells import CutCell,CutFace,CutCellMesh
    from braids_core.boundaries.inlet_io import NativeGrid
    meshes=[]
    for mesh in value['meshes']:
        cells=tuple(CutCell(**{k:_tuples(v) for k,v in c.items()}) for c in mesh['cells'])
        faces=tuple(CutFace(**{k:_tuples(v) for k,v in f.items()}) for f in mesh['faces'])
        if any(c.cell_id!=i or c.area_m2<=0 for i,c in enumerate(cells)):raise ValueError('Cached cell geometry differs')
        if any(f.face_id!=i or not 0<=f.owner<len(cells) or not -1<=f.neighbour<len(cells) or f.length_m<=0 for i,f in enumerate(faces)):
            raise ValueError('Cached face geometry differs')
        meshes.append(CutCellMesh(mesh['domain_id'],mesh['crs_wkt'],NativeGrid(**mesh['grid']),cells,faces,tuple(mesh['obstacle_ids']),mesh['solid_area_m2']))
    arrays={k:np.asarray(value[k],dtype=np.int64) for k in ('parent','node_level','maximum_level')}
    count=sum(len(m.cells) for m in meshes)
    if any(v.shape!=(count,) for v in arrays.values()) or len(value['children'])!=count or value['offsets'][-1]!=count:raise ValueError('Cached hierarchy shape differs')
    for i,children in enumerate(value['children']):
        if any(not 0<=child<count or arrays['parent'][child]!=i for child in children):raise ValueError('Cached ancestry differs')
    return AMRHierarchy(tuple(meshes),tuple(value['offsets']),arrays['parent'],_tuples(value['children']),arrays['node_level'],arrays['maximum_level'],value['refinement_ratio'])


def cached(root,contract,filename,build,*,stop):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);key=identity(contract);target=root/key
    worker.check_stop(stop)
    if target.exists():
        receipt=json.loads((target/'ASSET.json').read_text(encoding='utf-8'))
        if receipt!={'contract':contract,'sha256':sha(target/filename)}:raise ValueError('Preparation asset checksum/contract differs: '+str(target))
        return target/filename,True
    with tempfile.TemporaryDirectory(prefix='.asset-',dir=root) as temporary:
        pending=Path(temporary)/'ready';pending.mkdir();build(pending/filename);worker.check_stop(stop)
        write_json_atomic(pending/'ASSET.json',dict(contract=contract,sha256=sha(pending/filename)))
        try:publish_directory(pending,target,lambda:worker.check_stop(stop))
        except FileExistsError:
            # Another Desktop process published this same content key first.
            # Never replace it: re-enter the normal checksum/contract gate.
            return cached(root,contract,filename,build,stop=stop)
    return target/filename,False


def prepare(project_path,output,*,asset_cache,stop_file=None):
    project_path=Path(project_path).resolve();project=load_project(project_path);target=Path(output)
    if target.exists():raise FileExistsError(target)
    target.mkdir(parents=True);write_json_atomic(target/'PROJECT.json',project)
    timings={};reuse={};start=time.perf_counter()
    def progress(phase,percent):
        worker.check_stop(stop_file);write_json_atomic(target/'PROGRESS.json',dict(status='preparing',phase=phase,percent=percent))
    def timed(name,fn):
        stamp=time.perf_counter();value=fn();timings[name]=time.perf_counter()-stamp;return value
    try:
        progress('Verifying preparation dependencies',1)
        runtime=public.source_identity();helper=sha(__file__);profile=worker_profile(project);_,material=worker.validate_profile(profile)
        dem=Path(project['terrain']['dem_path']);dem=dem if dem.is_absolute() else project_path.parent/dem
        from braids_core.river_folder import _raster_files
        dependencies={p.name:sha(p) for p in _raster_files(dem)}
        geometry=dict(schema=1,runtime=runtime,helper=helper,**{k:profile[k] for k in ('domain','assignments','grid','levels','breaklines','solids','local_refinement')})
        progress('Loading or constructing mesh geometry',3)
        built={}
        def build_tree(path):
            built['tree']=worker.hierarchy(profile,stop_file);write_json_atomic(path,encode_tree(built['tree']))
        path,reuse['geometry']=timed('geometry_s',lambda:cached(Path(asset_cache)/'geometry',geometry,'TREE.json',build_tree,stop=stop_file))
        tree=built.get('tree')
        if tree is None:tree=timed('geometry_decode_s',lambda:decode_tree(json.loads(path.read_text(encoding='utf-8'))))
        progress('Reusing DEM integrals' if reuse['geometry'] else 'Integrating DEM',20)
        terrain_contract=dict(schema=1,geometry=identity(geometry),dem=dependencies)
        def build_terrain(path):
            spec=importlib.util.spec_from_file_location('_desktop_surface',worker.ROOT/'braids_qgis/core/raster_surface.py')
            surface=importlib.util.module_from_spec(spec);spec.loader.exec_module(surface)
            z,audit=surface.surface_averages(tree.finest,dem,progress=lambda v:progress('Integrating DEM',20+.3*v),
                cancelled=lambda:stop_file is not None and Path(stop_file).exists(),threads=profile['controls']['threads'])
            save_arrays_atomic(path,z=z,audit=np.array(json.dumps(audit)))
        path,reuse['terrain']=timed('terrain_s',lambda:cached(Path(asset_cache)/'terrain',terrain_contract,'TERRAIN.npz',build_terrain,stop=stop_file))
        with np.load(path,allow_pickle=False) as f:z=f['z'].copy();audit=json.loads(str(f['audit']))
        initial=deepcopy(project['initial']);selected=project['model']['physics']['sediment']
        if selected is not None:
            from braids_core.sediment.bed_columns_v101 import ColumnMethodsV101
            from braids_core.sediment.river_material import RiverMaterial
            initial['layer_fractions']=RiverMaterial.from_columns(ColumnMethodsV101(**project['settings']['columns'])).input_column_fractions(initial['layer_fractions'],selected['fraction_basis']).tolist()
        # The same field recipe as adaptive_river_gis, always freshly resolved.
        thickness=np.asarray(initial['layer_thickness_m'],float);fractions=np.asarray(initial['layer_fractions'],float)
        h=np.maximum(float(initial['stage_m'])-z,0.);velocity=np.asarray(initial['velocity_m_s'],float)
        q=np.vstack((h,velocity[:,None]*h));q[1:,h<=1e-6]=0.;ng=len(material.fraction_ids);n=len(z)
        concentration=np.broadcast_to(np.asarray(initial['concentration_kg_m3'],float)[:,None],(ng,n)).copy();concentration[:,h<=1e-6]=0.
        fields=dict(q=q,rock=z-thickness.sum(),thickness=np.broadcast_to(thickness[:,None],(len(thickness),n)).copy(),
            fractions=np.broadcast_to(fractions[:,:,None],(len(thickness),ng,n)).copy(),concentration=concentration)
        from braids_core.river_material_zones import apply_initial
        apply_initial(project,tree,fields)
        worker.arrays_valid(fields,tree,material);progress('Mapping banks and normal-depth sections',53)
        from braids_core.river_bank_survey import compile_survey
        survey=None if project['bank_survey'] is None else compile_survey(project['bank_survey'],tree,cancelled=lambda:stop_file is not None and Path(stop_file).exists()).document()
        write_json_atomic(target/'BANK_SURVEY.json',survey)
        from braids_core.river_normal_slope import prepare_slopes
        write_json_atomic(target/'NORMAL_SLOPES.json',prepare_slopes(project['settings']['hydraulics'],tree.finest,fields['rock']+fields['thickness'].sum(0)))
        core=target/'core';prepared=core/'prepared';prepared.mkdir(parents=True)
        write_json_atomic(prepared/'profile.json',profile);save_arrays_atomic(prepared/'initial.npz',**fields)
        required=() if survey is None else tuple(survey['mapping']['required_leaves'])
        from braids_core.river_turbines import prepare as prepare_turbines
        turbines=prepare_turbines(project['turbines'],tree,fields['rock']+fields['thickness'].sum(0),cancelled=lambda:stop_file is not None and Path(stop_file).exists())
        write_json_atomic(target/'TURBINES.json',turbines)
        if turbines is not None:required=tuple(sorted(set(required)|set(turbines['required_leaves'])))
        progress('Preparing current native records and forcing',60)
        def native():
            cache=worker.cache_for(prepared/'cache',profile,fields,tree,stop_file)
            leaves=worker.initial_leaves(profile,fields,tree,required);mesh=tree.select(leaves)
            cache.get(leaves,mesh)
            # Display data only: avoid rebuilding polygon topology to show the
            # mesh that was just prepared. The core never reads this artifact.
            polygons=[np.asarray(c.vertices_xy,float) for c in mesh.cells]
            save_arrays_atomic(target/'MESH_PREVIEW.npz',vertices=np.concatenate(polygons),
                offsets=np.r_[0,np.cumsum([len(p) for p in polygons])],faces=np.array(len(mesh.faces)))
            return len(leaves)
        cells=timed('native_records_s',native)
        write_json_atomic(prepared/'PREPARED.json',dict(schema=worker.SCHEMA,worker_sha256=sha(worker.__file__),
            files={name:sha(prepared/name) for name in ('profile.json','initial.npz')},cells=cells,QGIS_admitted=False))
        write_json_atomic(core/'TERRAIN.json',audit);write_json_atomic(core/'RECIPE.json',dict(profile=profile,dem_path=str(dem),initial=initial))
        write_json_atomic(core/'PROGRESS.json',dict(status='prepared',phase='prepared',percent=100))
        if dependencies!={p.name:sha(p) for p in _raster_files(dem)} or runtime!=public.source_identity():raise ValueError('Preparation inputs/runtime changed while preparing')
        worker.check_stop(stop_file)
        write_json_atomic(target/'PREPARED.json',dict(schema=public.PREPARED_SCHEMA,project_id=identity(project),runtime_id=runtime,
            terrain_sha256=sha(dem),files={name:sha(target/name) for name in public.PREPARED_FILES}))
        timings['total_s']=time.perf_counter()-start
        write_json_atomic(target/'DESKTOP_PREPARATION.json',dict(schema=1,helper_sha256=helper,reused=reuse,timings=timings,cells=cells,
            preview_sha256=sha(target/'MESH_PREVIEW.npz')))
        progress('Prepared — geometry/terrain reused' if all(reuse.values()) else 'Prepared',100);return target
    except BaseException as exc:
        write_json_atomic(target/'STATUS.json',dict(status='cancelled' if isinstance(exc,InterruptedError) else 'failed',error=f'{type(exc).__name__}: {exc}'))
        raise
