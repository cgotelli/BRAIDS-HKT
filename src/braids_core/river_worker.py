"""One strict public prepare/run/restart entry for the tab-based river product.

The common adaptive numerical owner is an internal library, not an alternative
product format. Public operations always require this project's full identity.
"""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import tempfile
from copy import deepcopy

import numpy as np

from .river_project import canonical, identity, load_project, read_json, worker_profile, fail
from .sediment import adaptive_river_worker as core
from .sediment.adaptive_river_gis import prepare_recipe
from .sediment.river_leaf_io import sha
from .windows_paths import filesystem_path
from braids_gpu.atomic_output import save_arrays_atomic, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
PREPARED_SCHEMA = "braids.river-prepared.v4"
CHECKPOINT_SCHEMA = "braids.river-checkpoint.v5"
PREPARED_FILES = ("PROJECT.json", "core/prepared/PREPARED.json",
                  "core/TERRAIN.json", "core/RECIPE.json", "BANK_SURVEY.json", "NORMAL_SLOPES.json", "TURBINES.json")


def _source_hashes():
    paths = set()
    for folder in ("braids_core", "braids_gpu", "solver/wave", "solver/cut_cells"):
        for p in (ROOT/folder).rglob("*"):
            if p.is_file() and p.suffix in (".py", ".h", ".cpp", ".cu", ".f90", ".f"):
                paths.add(p)
    # These implementations are still imported by the maintained owner.
    # Packaging must retain or relocate them, never omit them from provenance.
    for folder in ("validation/sediment/river_morphodynamics", "validation/gpu_fwave"):
        paths.update(p for p in (ROOT/folder).glob("*") if p.suffix in (".py", ".h", ".cu") and p.is_file())
    paths.add(ROOT/"braids_qgis/core/raster_surface.py")
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def source_identity():
    """Bind the public contract and maintained source tree, not an old release."""
    return identity(_source_hashes())


def _retry202_legacy_runtime_ids():
    """Recognize only the exact predecessor tree for the clock migration.

    All other source files must match; do not accept a generic older runtime.
    """
    hashes=_source_hashes()
    performance='braids_core/sediment/runtime_performance.py'
    if hashes.pop(performance,None) not in ('8532128f0b31b0edcd57551d8521b7c6527e2182afb3177a6baf38139e370263','d9dd5c3157351be1a491d952152de8d5f10377853d4f712c5dfcb35a165bbc01'):
        return set()
    required={              'braids_core/sediment/runtime_shallow_fusion.py':'c4e4c33d9d8c7f2e92145f5dd72a195d6ac6609dbbf165db93488bf7b4b40413',
              'braids_core/sediment/runtime_topology_cache.py':'e7f5d494e80195aa43deff9ab5e78418fab3a105a1257f8ea5e700da51376e9a',
              'braids_core/sediment/runtime_snapshot.py':'19cc72f3ac232c9b99d3cae79dbe40c74f77fcf7238bdf36ff82d88d5ac9c9bd'}
    if any(hashes.pop(name,None)!=value for name,value in required.items()):return set()
    field='braids_core/sediment/river_field_policy.py'
    clock='braids_core/sediment/river_sediment_clock.py'
    worker='braids_core/river_worker.py'
    if hashes.get(clock)!='08a21cdb74e7a4eff4df2a6ccd61c3e7a28cadee8edfcf8475f8376e25b9dedb':return set()
    variants=[
        ('77c79f1992773374259a1b7cb66d8d6891041653e8b1a84da6a41d36044a7cfe',
         '4c215303f82512d28e47d9d794d134cb00c699d58380c9e6a7286a76121dcf77',
         '47d71225c04b297f53070a88fc4c9be54fb5ed57dc66600682418425cbd0c534'),
        ('79e3275300151160ca58d2174f33f65f638f938858fb1d39f2a50c387537d962',
         'bde8d175361ba1ad00edb33a6728c1d200d24aa38a210ef5c7ee80a25b69893e',
         'fa356e9b4a4a7f6bc7157c19febd69f0c715c0839ba951d65570e69b2ffdbc6c')]
    old_clock='78c8d04c9e0e19e49826e74bbdd1af4628b0b81f0d71e98464ef3396c6b969ab'
    result={identity(dict(hashes,**{field:old_field,clock:old_clock,worker:old_worker}))
            for current_field,old_field,old_worker in variants if hashes.get(field)==current_field}
    worker202={
        '77c79f1992773374259a1b7cb66d8d6891041653e8b1a84da6a41d36044a7cfe':'53f6d306bded21b7e17c031b234da218a7c5be0daaab1f976a82c9e181db378b',
        '79e3275300151160ca58d2174f33f65f638f938858fb1d39f2a50c387537d962':'de47b4c3aab220da838a5174dbef49951b94f712dd6f51f1f7d318068dd1743b'}
    if hashes.get(field) in worker202:
        result.add(identity(dict(hashes,**{worker:worker202[hashes[field]]})))
    worker204={
        '77c79f1992773374259a1b7cb66d8d6891041653e8b1a84da6a41d36044a7cfe':'820c80bfa11c0c0667a641c7ce38e75ab8ff608be08edbbd11ff04acf6f1d0ac',
        '79e3275300151160ca58d2174f33f65f638f938858fb1d39f2a50c387537d962':'ffcd8270d5aec854773b7c3ad5b536a5a5e23809e6fde8ded03545078689280d'}
    if hashes.get(field) in worker204:
        result.add(identity(dict(hashes,**{worker:worker204[hashes[field]],performance:'9122e6fcb024697f28b9d507e9034d169b7166ba573fbc365ae5168de054e4ab'})))
    worker205={
        '77c79f1992773374259a1b7cb66d8d6891041653e8b1a84da6a41d36044a7cfe':'49680d75f222b3d1e755c12169efb7c58c05e4407af07f5bd7ab6aec6321d577',
        '79e3275300151160ca58d2174f33f65f638f938858fb1d39f2a50c387537d962':'a571945878cf82c55b05a02067e4955461bcbe49e8ac6b2e3e6770d35b60a53c'}
    if hashes.get(field) in worker205:
        result.add(identity(dict(hashes,**{worker:worker205[hashes[field]],performance:'2103545cd4c91f08ed1ab8dca6a48dba962d9d5198c8f40b723a8ae1013aafdd'})))
    worker206={
        '77c79f1992773374259a1b7cb66d8d6891041653e8b1a84da6a41d36044a7cfe':'760832a0c929084a5fa979676d4c3368abd8ab44dfedb2f2cd437f350f51cf88',
        '79e3275300151160ca58d2174f33f65f638f938858fb1d39f2a50c387537d962':'0d3252213520447b746ede54027b18d7800dfa7b152d514cd7d1c673789b11c8'}
    if hashes.get(field) in worker206:
        result.add(identity(dict(hashes,**{worker:worker206[hashes[field]],performance:'85b3a5bbdf3fa90ebc2bca107a6b5f486fa84c24d0c2a5212e5b84576a4e6663',
            'braids_core/sediment/runtime_snapshot.py':'19cc72f3ac232c9b99d3cae79dbe40c74f77fcf7238bdf36ff82d88d5ac9c9bd'})))
    worker207={
        '77c79f1992773374259a1b7cb66d8d6891041653e8b1a84da6a41d36044a7cfe':'5e26322d8c903e52ccb71defc67e80d08da6975edf34167c222e3734b56b88cf',
        '79e3275300151160ca58d2174f33f65f638f938858fb1d39f2a50c387537d962':'dda5f67bff9c2914b60247f15a486ecc22122c7736758686f8a46fdbf6bb25f2'}
    if hashes.get(field) in worker207:
        result.add(identity(dict(hashes,**{worker:worker207[hashes[field]],performance:'19976d9c63b1d54bb898cda78502038da975c0f53a1864e598b27267af2ef902',
            'braids_core/sediment/runtime_snapshot.py':'19cc72f3ac232c9b99d3cae79dbe40c74f77fcf7238bdf36ff82d88d5ac9c9bd',
            'braids_core/sediment/runtime_topology_cache.py':'e7f5d494e80195aa43deff9ab5e78418fab3a105a1257f8ea5e700da51376e9a'})))
    return result


def prepare(project_path, output, *, stop_file=None):
    """Reject formats/physics before creating folders or constructing a mesh."""
    project_path = filesystem_path(project_path).resolve()
    project = load_project(project_path)
    core.check_stop(stop_file)
    target = filesystem_path(output)
    if target.exists():
        raise FileExistsError(target)
    dem = Path(project["terrain"]["dem_path"])
    if not dem.is_absolute():
        dem = project_path.parent/dem
    dem = filesystem_path(dem).resolve()
    if not dem.is_file():
        raise FileNotFoundError(dem)
    terrain_sha = sha(dem)
    runtime = source_identity()
    target.mkdir(parents=True)
    write_json_atomic(target/"PROJECT.json", project)
    def progress(phase, percent):
        write_json_atomic(target/"PROGRESS.json", dict(status="preparing", phase=phase, percent=percent))
    try:
        initial = deepcopy(project["initial"])
        selected = project["model"]["physics"]["sediment"]
        if selected is not None:
            from .sediment.bed_columns_v101 import ColumnMethodsV101
            from .sediment.river_material import RiverMaterial
            material = RiverMaterial.from_columns(ColumnMethodsV101(**project["settings"]["columns"]))
            initial["layer_fractions"] = material.input_column_fractions(
                initial["layer_fractions"], selected["fraction_basis"]).tolist()
        recipe = dict(profile=worker_profile(project), dem_path=str(dem), initial=initial)
        turbine_map=[None]
        def survey(tree):
            from .river_bank_survey import compile_survey
            mapped=(None if project['bank_survey'] is None else compile_survey(project['bank_survey'],tree,
                cancelled=lambda:stop_file is not None and filesystem_path(stop_file).exists()).document())
            write_json_atomic(target/'BANK_SURVEY.json',mapped)
            from .river_turbines import prepare as prepare_turbines
            turbine_map[0]=prepare_turbines(project['turbines'],tree,np.zeros(len(tree.finest.cells)),cancelled=lambda:stop_file is not None and filesystem_path(stop_file).exists())
            pins=() if turbine_map[0] is None else tuple(turbine_map[0]['required_leaves'])
            return pins+(() if mapped is None else tuple(mapped['mapping']['required_leaves']))
        def normal_slopes(tree,fields):
            from .river_material_zones import apply_initial
            apply_initial(project,tree,fields)
            from .river_normal_slope import prepare_slopes
            progress('normal-depth mesh slope',55)
            mapped=prepare_slopes(project['settings']['hydraulics'],tree.finest,fields['rock']+fields['thickness'].sum(0))
            write_json_atomic(target/'NORMAL_SLOPES.json',mapped)
            if turbine_map[0] is not None:
                bed=fields['rock']+fields['thickness'].sum(0)
                for rotor in turbine_map[0]['rotors']:
                    rotor['hub_elevation_m']=float(max(bed[rotor['hub_reference_finest_cells']])+rotor['hub_height_m'])
            write_json_atomic(target/'TURBINES.json',turbine_map[0])
        prepare_recipe(recipe, target/"core", stop_file, progress_callback=progress,on_hierarchy=survey,on_fields=normal_slopes)
        core.check_stop(stop_file)
        if sha(dem) != terrain_sha:
            raise ValueError("DEM changed during preparation")
        if source_identity() != runtime:
            raise ValueError("Runtime changed during preparation")
        receipt = dict(schema=PREPARED_SCHEMA, project_id=identity(project),
                       runtime_id=runtime, terrain_sha256=terrain_sha,
                       files={name: sha(target/name) for name in PREPARED_FILES})
        write_json_atomic(target/"PREPARED.json", receipt)
        write_json_atomic(target/"PROGRESS.json", dict(status="prepared", phase="prepared", percent=100))
        return target
    except BaseException as exc:
        write_json_atomic(target/"STATUS.json", dict(
            status="cancelled" if isinstance(exc, InterruptedError) else "failed",
            error=f"{type(exc).__name__}: {exc}"))
        raise


def load_prepared(path):
    """No raw carrier folders, old profiles or missing-identity fallbacks."""
    root = filesystem_path(path)
    receipt_path = root/"PREPARED.json"
    if not receipt_path.is_file():
        fail("prepared", "A current prepared river project is required", "unsupported_format")
    receipt = read_json(receipt_path.read_text(encoding="utf-8"))
    keys = {"schema", "project_id", "runtime_id", "terrain_sha256", "files"}
    if not isinstance(receipt, dict) or receipt.get("schema") != PREPARED_SCHEMA or set(receipt) != keys:
        fail("prepared.schema", "Old prepared runs are not supported; prepare a new river project",
             "unsupported_format")
    if not isinstance(receipt["files"], dict) or set(receipt["files"]) not in (set(PREPARED_FILES), set(PREPARED_FILES) | {"TOPOLOGY.json"}):
        fail("prepared.files", "Prepared file inventory differs", "identity_mismatch")
    if receipt["runtime_id"] != source_identity() and receipt["runtime_id"] not in _retry202_legacy_runtime_ids():
        fail("prepared.runtime_id", "Runtime changed; prepare again explicitly", "identity_mismatch")
    for name, digest in receipt["files"].items():
        path = root/name
        if not path.is_file() or sha(path) != digest:
            fail("prepared.files", "Prepared artifact changed: " + name, "identity_mismatch")
    project = load_project(root/"PROJECT.json")
    if identity(project) != receipt["project_id"]:
        fail("prepared.project_id", "Project identity differs", "identity_mismatch")
    turbine_map=read_json((root/'TURBINES.json').read_text(encoding='utf-8'))
    if ((project['turbines'] is None and turbine_map is not None) or
        (project['turbines'] is not None and (not isinstance(turbine_map,dict) or turbine_map.get('definition')!=project['turbines']))):
        fail('prepared.turbines','Prepared turbine definition differs; prepare again.','identity_mismatch')
    survey=read_json((root/'BANK_SURVEY.json').read_text(encoding='utf-8'))
    if project['bank_survey'] is None:
        if survey is not None:fail('bank_survey','Inactive prepared survey is nonempty','identity_mismatch')
    else:
        from .river_bank_survey import BankSurveyMap
        BankSurveyMap.from_document(survey,project['bank_survey'])
    from .river_normal_slope import resolved_hydraulics
    resolved_hydraulics(project['settings']['hydraulics'],read_json((root/'NORMAL_SLOPES.json').read_text(encoding='utf-8')))
    # The internally generated input cannot diverge from the public project.
    expected = worker_profile(project)
    stored = read_json((root/"core/prepared/profile.json").read_text(encoding="utf-8"))
    if canonical(stored) != canonical(expected):
        fail("prepared.profile", "Numerical inputs differ from the public project", "identity_mismatch")
    return project, receipt


def checkpoint_identity(project, receipt):
    return dict(schema=CHECKPOINT_SCHEMA, project_id=identity(project),
                prepared_id=identity(receipt), runtime_id=receipt["runtime_id"],
                model_id=project["model"]["configuration_id"],
                backend=project["model"]["request"]["execution"]["backend"])


def checkpoint_identities(project, receipt):
    expected=checkpoint_identity(project,receipt)
    legacy=_retry202_legacy_runtime_ids()
    if receipt['runtime_id'] in legacy:
        current=source_identity()
        prior=sorted(legacy-{receipt['runtime_id'],current})
        return (expected,*(dict(expected,runtime_id=value) for value in prior),dict(expected,runtime_id=current))
    return (expected,)


def publish_checkpoint(source, target, expected,*,state,initialization=b''):
    payload = filesystem_path(source).read_bytes()
    metadata = dict(expected, payload_sha256=hashlib.sha256(payload).hexdigest(),
        state_sha256=hashlib.sha256(state).hexdigest(),initialization_sha256=hashlib.sha256(initialization).hexdigest())
    save_arrays_atomic(target, identity=np.array(canonical(metadata)),
                       payload=np.frombuffer(payload, dtype=np.uint8),state=np.frombuffer(state,dtype=np.uint8),
                       initialization=np.frombuffer(initialization,dtype=np.uint8))


def read_checkpoint(path, expected):
    """Authenticate the outer format before constructing any numerical owner."""
    return read_checkpoint_bundle(path,expected)['payload']


def read_checkpoint_bundle(path,expected):
    from .river_state_snapshot import archive,sha_bytes,decode
    saved=archive(filesystem_path(path).read_bytes())
    if set(saved)!={'identity','payload','state','initialization'}:
        fail('checkpoint','Current exact-restart format required. Older accepted fields may be loaded through saved-state initialization.','unsupported_format')
    if saved['identity'].shape!=() or any(saved[k].dtype!=np.uint8 or saved[k].ndim!=1 for k in ('payload','state','initialization')):
        fail('checkpoint','Checkpoint field types differ','identity_mismatch')
    metadata=read_json(str(saved['identity']));payload=saved['payload'].tobytes();state=saved['state'].tobytes();initialization=saved['initialization'].tobytes()
    if not isinstance(metadata, dict) or metadata.get("schema") != CHECKPOINT_SCHEMA:
        fail("checkpoint.schema", "Current river checkpoint format required", "unsupported_format")
    checksums=dict(payload_sha256=sha_bytes(payload),state_sha256=sha_bytes(state),initialization_sha256=sha_bytes(initialization))
    if metadata != dict(expected,**checksums):
        legacy=_retry202_legacy_runtime_ids()
        compatible=(expected['runtime_id'] in legacy and
                    any(metadata==dict(expected,runtime_id=value,**checksums)
                        for value in legacy|{source_identity()}))
        if not compatible:
            fail("checkpoint", "Checkpoint project, physics, backend, runtime or payload differs",
                 "identity_mismatch")
    decode(state)
    return dict(payload=payload,state=state,initialization=initialization)


def validate_continuation(original,current):
    """Only extend the clock; never reinterpret a saved physical state."""
    from .river_project import validate_project
    validate_project(current)
    def physical(doc):
        doc=deepcopy(doc);doc.pop('editor_state',None);doc.pop('name',None)
        doc['settings'].pop('end_time_s');doc['settings'].pop('checkpoint_interval_s')
        return doc
    if canonical(physical(original))!=canonical(physical(current)):
        fail('checkpoint','Continuation permits only end time and checkpoint cadence changes. Restore the original physics, mesh, forcing, backend and inputs, or start a new run.')
    if current['settings']['end_time_s']<original['settings']['end_time_s']:fail('checkpoint','Continuation cannot shorten the original run horizon')
    return current['settings']['end_time_s']


def run(prepared, output, *, restart=None, until=None, stop_file=None, on_frame=None,checkpoint_dir=None,continuation_project=None,initial_state=None):
    project, receipt = load_prepared(prepared)
    if project.get('power_plan') and (restart is not None or initial_state is not None or continuation_project is not None):
        raise ValueError('Power planning requires a fresh turbine-free reference run. Imported/restarted fields are not yet certified as undisturbed; select a fresh run.')
    identities = checkpoint_identities(project, receipt)
    expected,save_identity=identities[0],identities[-1]
    from .river_normal_slope import resolved_hydraulics
    normal_slopes=read_json((filesystem_path(prepared)/'NORMAL_SLOPES.json').read_text(encoding='utf-8'))
    hydraulics=resolved_hydraulics(project['settings']['hydraulics'],normal_slopes)
    # This check deliberately precedes output creation and core.run().
    bundle=None if restart is None else read_checkpoint_bundle(restart,expected)
    payload=None if bundle is None else bundle['payload']
    initializer=b'' if initial_state is None else filesystem_path(initial_state).read_bytes()
    if bundle is not None:
        if initializer and initializer!=bundle['initialization']:fail('checkpoint','Exact restart cannot replace its initialization')
        initializer=bundle['initialization']
    from .river_state_initialization import load as load_initial_state
    initialized=None if not initializer else load_initial_state(initializer,project,receipt)
    turbine_map=read_json((filesystem_path(prepared)/'TURBINES.json').read_text(encoding='utf-8'))
    if initialized is not None:
        normal_slopes=initialized[0]['normal_slopes']
        hydraulics=resolved_hydraulics(project['settings']['hydraulics'],normal_slopes)
        if turbine_map is not None:
            for rotor in turbine_map['rotors']:rotor['hub_elevation_m']=initialized[0]['turbine_hub_elevations_m'][rotor['name']]
    target_project=project
    if continuation_project is not None:
        if restart is None:fail('checkpoint','A continuation requires a selected checkpoint')
        target_project=load_project(continuation_project)
        until=validate_continuation(project,target_project)
    horizon=project['settings']['end_time_s'] if until is None else until
    if horizon>project['settings']['end_time_s']:
        if restart is None:fail('until','Extending a run requires an authenticated checkpoint')
        extended=deepcopy(project);extended['settings']['end_time_s']=horizon
        extended.pop('editor_state',None)
        from .river_project import validate_project
        validate_project(extended) # Every hydraulic/sediment table must cover it.
    core.check_stop(stop_file)
    output = filesystem_path(output)
    if output.exists():
        raise FileExistsError(output)
    if until is not None and (type(until) not in (int, float) or not np.isfinite(until)
                             or not 0 < until):
        fail("until", "Segment target must be positive finite seconds")
    raw = output/"internal/checkpoint.npz"
    # Keep the relationship between a run and its immutable preparation when
    # the containing project folder is moved. Cross-drive CLI runs are still
    # explicit external references; folder-managed runs always share a root.
    try:
        input_reference = Path(os.path.relpath(filesystem_path(prepared).resolve(), output.resolve())).as_posix()
        input_base = "run_directory"
    except ValueError:
        input_reference = str(filesystem_path(prepared).resolve())
        input_base = "absolute"
    context = dict(schema="braids.river-results.v1", project_id=identity(project),
                   execution_end_time_s=horizon,continued_from=str(restart) if restart is not None else None,
                   model=project["model"], input=input_reference, input_path_base=input_base,
                   project_file="PROJECT.json",
                   boundaries=dict(hydraulics=project['settings']['hydraulics'],states=project['boundary_states'],
                       resolved=project['boundary_resolution'],normal_slopes=normal_slopes),
                   roughness_parameters=project["roughness_parameters"],
                   skin_shear_parameters=project["skin_shear_parameters"],
                   wet_avalanche_parameters=project["wet_avalanche_parameters"],
                   bank_survey=project['bank_survey'],
                   sediment_forcing=dict(fraction_ids=project["settings"]["columns"]["fraction_ids"],
                       boundaries=project["settings"]["sediment_boundaries"],
                       schedules=project["sediment_schedules"],
                       initial_concentration_kg_m3=project["initial"]["concentration_kg_m3"]),
                   dependent_layer_parameters=project["dependent_layer_parameters"],
                   turbines=project['turbines'],crs=project["settings"]["domain"]["spatial_reference"],
                   runtime_id=save_identity['runtime_id'],prepared_runtime_id=receipt["runtime_id"])
    if initialized is not None:context['state_initialization']=initialized[0]
    def created():
        # A result keeps the exact resolved parameters used for that run,
        # independently of later edits to the project's current revision.
        write_json_atomic(output/"PROJECT.json", project)
        context["project_sha256"] = sha(output/"PROJECT.json")
        write_json_atomic(output/"RUN_CONTEXT.json", context)
        if initializer:
            from braids_gpu.atomic_output import write_bytes_atomic
            write_bytes_atomic(output/'INITIAL_STATE.npz',initializer)
    def published(folder, row):
        if on_frame is not None:
            on_frame(folder, row)
    def checkpointed(path, row,owner,mesh):
        # Called only after an atomic raw checkpoint publication, independently
        # of the solution-frame clock. The core retries a failed notification
        # before returning, without rewriting the accepted raw state.
        from .river_state_snapshot import capture
        from .river_jiang_cao import active
        state=capture(project,mesh,owner,jiang_cao=active(project['model']))
        publish_checkpoint(path, output/"checkpoint.npz", save_identity,state=state,initialization=initializer)
        if checkpoint_dir is not None:
            folder=filesystem_path(checkpoint_dir);folder.mkdir(parents=True,exist_ok=True)
            target=folder/f"t_{row['time_s']:.12g}_step_{row['steps']:08d}.npz"
            publish_checkpoint(path,target,save_identity,state=state,initialization=initializer)
            write_json_atomic(target.with_suffix('.json'),dict(schema='braids.checkpoint-location.v1',
                prepared=os.path.relpath(filesystem_path(prepared).resolve(),folder.resolve()),
                run=os.path.relpath(output.resolve(),folder.resolve()),sha256=sha(target),**row))
    with tempfile.TemporaryDirectory(prefix="braids-current-restart-") as temp:
        restart_path = None
        if payload is not None:
            restart_path = Path(temp)/"owner.npz"
            restart_path.write_bytes(payload)
        from .river_boundary import carrier_schedules
        from .river_jiang_cao import active as jc_active
        if jc_active(project['model']):
            from .sediment.jiang_cao_desktop import run as run_jc
            return run_jc(filesystem_path(prepared)/'core/prepared', output,
                project=project, receipt=receipt,
                boundaries=dict(hydraulics=hydraulics,states=project['boundary_states'],
                                resolved=project['boundary_resolution'],normal_slopes=normal_slopes),
                restart=restart_path,until=until,stop_file=stop_file,on_frame=published,
                checkpoint_path=raw,on_created=created,on_checkpoint_state=checkpointed,initial_state=initialized,
                checkpoint_interval_s=target_project['settings']['checkpoint_interval_s'])
        return core.run(filesystem_path(prepared)/"core/prepared", output,
                            backend=expected["backend"], restart=restart_path, until=until,
                            stop_file=stop_file, on_frame=published,
                            resolved_model=project["model"], checkpoint_path=raw, on_created=created,
                            checkpoint_interval_s=target_project['settings']['checkpoint_interval_s'],
                            on_checkpoint_state=checkpointed,initial_state=initialized,
                            process_parameters=dict(turbines=turbine_map,slope=project["slope_parameters"],
                                                    bank_survey=read_json((filesystem_path(prepared)/'BANK_SURVEY.json').read_text(encoding='utf-8')),
                                                    boundaries=dict(hydraulics=hydraulics,
                                                        states=project['boundary_states'],resolved=project['boundary_resolution'],normal_slopes=normal_slopes),
                                                    sediment_schedules=carrier_schedules(project['sediment_schedules'],
                                                        project['settings']['sediment_boundaries'],project['settings']['columns']),
                                                    transport_boundaries=project['settings']['sediment_boundaries'],
                                                    hiding=project["hiding_parameters"],
                                                    roughness=project["roughness_parameters"],
                                                    material_zones=project['material_zones'],
                                                    skin_shear=project["skin_shear_parameters"],
                                                    wet_avalanche=project["wet_avalanche_parameters"],
                                                    van_rijn_reference=project["van_rijn_reference_parameters"],
                                                    garcia_parker=project.get('garcia_parker_parameters'),
                                                    de_leeuw=project.get('de_leeuw_parameters'),
                                                    supplied_suspension=project.get('supplied_suspension_parameters'),
                                                    classic_suspension=project.get('classic_suspension_parameters'),
                                                    dependent_layers=project["dependent_layer_parameters"],
                                                    output_fields=project['settings'].get('output_fields')))


_ordinary_run = run


@functools.wraps(_ordinary_run)
def run(*args, **kwargs):
    from .sediment.runtime_performance import optimized_river_run
    prepared=args[0] if args else kwargs.get('prepared')
    with optimized_river_run(prepared=prepared,restart=kwargs.get('restart')):
        return _ordinary_run(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="BRAIDS unified explicit f-wave river solver")
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--project", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stop-file")
    p = sub.add_parser("run")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--restart")
    p.add_argument("--until", type=float)
    p.add_argument("--stop-file")
    p.add_argument('--checkpoint-dir')
    p.add_argument('--continuation-project')
    p.add_argument('--initial-state')
    args = parser.parse_args()
    if args.action == "prepare":
        print(prepare(args.project, args.output, stop_file=args.stop_file))
    else:
        print(json.dumps(run(args.input, args.output, restart=args.restart, until=args.until,
                             stop_file=args.stop_file,checkpoint_dir=args.checkpoint_dir,continuation_project=args.continuation_project,initial_state=args.initial_state)))


if __name__ == "__main__":
    main()
