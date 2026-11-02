"""Paired 24 s cold-mobile CPU sensitivity diagnostic, using a locked binary.

No compilation, restart, numerical control changes or GPU execution. One case
repeats the reference prefix; the other changes each nonzero initial hu/hv by
exactly one nextafter step towards positive infinity. No long-run qualification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TAG = 'cpu_cold_ulp24_20261002_v1'
SOURCE = ROOT/'braids_support/sediment/lfm_bend_2d/runs/fast_cpu_cold20h_20261001_v1_mobile_braids'
RUNTIME = ROOT/'braids_support/sediment/lfm_bend_2d/runs/fast_cpu_default_20261001_v1/runtime_mf25'
EXE_SHA = '9325aae9e8acdbbe9ed091044a17e847bc04881bd924e83d137565947b07c590'
MAGIC = b'BRAIDS_NATIVE_V1'
OLD_TIME = '2880 0.012500000000000001 24'
NEW_TIME = '24 0.012500000000000001 24'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition: raise ValueError(message)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def decode_binary(blob):
    require(blob[:16] == MAGIC and len(blob) >= 52, 'Unsupported native binary header')
    header = np.frombuffer(blob, dtype='<i4', count=9, offset=16).copy()
    version, nc, nf, ns = map(int, header[:4])
    require(version == 1 and nc > 0 and nf > 0 and ns == 9 and header[8] == 0,
            'Expected fixed-grid nine-physical-state binary layout')
    q_offset = 52 + 40*nc
    require(len(blob) >= q_offset + 8*nc*ns, 'Truncated native state')
    q = np.frombuffer(blob, dtype='<f8', count=nc*ns, offset=q_offset).reshape(nc, ns).copy()
    area = np.frombuffer(blob, dtype='<f8', count=nc, offset=52+8*nc).copy()
    xy = np.frombuffer(blob, dtype='<f8', count=2*nc, offset=52+16*nc).reshape(nc, 2).copy()
    rock = np.frombuffer(blob, dtype='<f8', count=nc, offset=52+32*nc).copy()
    require(np.isfinite(q).all() and np.all(area > 0), 'Invalid initial physical arrays')
    return dict(header=header, q=q, area=area, xy=xy, rock=rock, q_offset=q_offset)


def perturb_binary(blob):
    data = decode_binary(blob); original = data['q']; changed = original.copy()
    for component in (1, 2):
        active = original[:, component] != 0.
        changed[active, component] = np.nextafter(original[active, component], np.inf)
    require(np.isfinite(changed).all(), 'Perturbation overflow')
    begin = data['q_offset']; end = begin + changed.nbytes
    output = blob[:begin] + changed.astype('<f8', copy=False).tobytes() + blob[end:]
    # Byte masks are independent of floating equality, retaining signed zeros.
    allowed = np.zeros(len(blob), dtype=bool)
    state_mask = np.zeros(changed.shape, dtype=bool)
    state_mask[:, 1:3] = original[:, 1:3] != 0.
    allowed[begin:end] = np.repeat(state_mask.ravel(), 8)
    left = np.frombuffer(blob, dtype=np.uint8); right = np.frombuffer(output, dtype=np.uint8)
    require(np.array_equal(left[~allowed], right[~allowed]), 'Changed an unauthorized binary byte')
    decoded = decode_binary(output)
    require(np.array_equal(decoded['q'].view(np.uint64), changed.view(np.uint64)), 'Binary state roundtrip failed')
    records = {}
    for component, name in ((1, 'hu'), (2, 'hv')):
        active = state_mask[:, component]; delta = changed[:, component] - original[:, component]
        records[name] = dict(changed_cells=int(active.sum()), maximum_absolute_delta_m2_s=float(abs(delta).max()),
            maximum_relative_delta=float(np.max(abs(delta[active]/original[active, component]), initial=0.)),
            area_integrated_delta_m4_s=float(delta @ data['area']),
            exact_nextafter_toward_positive_infinity=True)
    return output, data, changed, dict(components=records, changed_bytes=int(np.sum(left != right)),
        unauthorized_bytes_unchanged=True, depth_bed_suspended_layers_and_zero_momenta_bitwise_unchanged=True,
        secondary_note='Physical binary has nine states. Prognostic hI initialization remains implicit zero in the unchanged v37 loader.')


def fingerprint(text, source_sha256):
    # This mirrors the public writer: omit fingerprint and final newline, and
    # normalize ONLY the end-time token. Embedded binary SHA remains included.
    require(text.endswith('\n') and '\r' not in text, 'Require original LF deck serialization')
    rows = text[:-1].split('\n'); old = rows.pop(1)
    matches = [i for i, row in enumerate(rows) if row in (OLD_TIME, NEW_TIME)]
    require(len(matches) == 1, 'Unexpected time-control record')
    rows[matches[0]] = 'END 0.012500000000000001 24'
    value = hashlib.sha256('\n'.join(rows).encode('utf-8'))
    value.update(json.dumps(source_sha256, sort_keys=True).encode())
    return value.hexdigest(), old


def transform_deck(text, metadata, binary_sha):
    expected, original = fingerprint(text, metadata['source_sha256'])
    require(expected == original == metadata['fingerprint'], 'Original writer fingerprint could not be reproduced')
    rows = text[:-1].split('\n'); before = rows.copy()
    times = [i for i, row in enumerate(rows) if row == OLD_TIME]
    binaries = [i for i, row in enumerate(rows) if row.startswith('case.data.bin ')]
    require(len(times) == len(binaries) == 1, 'Ambiguous original input records')
    rows[times[0]] = NEW_TIME
    rows[binaries[0]] = 'case.data.bin ' + binary_sha
    trial = '\n'.join(rows)+'\n'
    current, _ = fingerprint(trial, metadata['source_sha256']); rows[1] = current
    permitted = {1, times[0], binaries[0]}
    require(all(a == b for i, (a, b) in enumerate(zip(before, rows)) if i not in permitted),
            'Unauthorized physical/control text change')
    return '\n'.join(rows)+'\n', dict(original_fingerprint=original, fingerprint=current,
        end_time_s=dict(original=2880., diagnostic=24.), unchanged_dt_ceiling_s=.0125,
        unchanged_output_interval_s=24., changed_line_indices_zero_based=[i for i, (a,b) in enumerate(zip(before,rows)) if a != b],
        every_other_text_record_bitwise_unchanged=True, end_time_normalized_by_original_writer=True)


def stage(destination, *, perturb):
    from .coupled_reference import copy_inputs
    from braids_support.sediment.lfm_bend_2d.cold_optimized_comparison import verify_inputs
    provenance = verify_inputs(SOURCE)
    origin = copy_inputs(SOURCE/'case.data', destination)
    original = (SOURCE/'case.data.bin').read_bytes()
    if perturb:
        binary, initial, changed, evidence = perturb_binary(original)
    else:
        binary = original; initial = decode_binary(original); changed = initial['q'].copy()
        evidence = dict(all_binary_bytes_identical_to_original=True)
    (destination/'case.data.bin').write_bytes(binary)
    metadata = json.loads((SOURCE/'case.data.json').read_text())
    text, controls = transform_deck((SOURCE/'case.data').read_bytes().decode(), metadata, sha(destination/'case.data.bin'))
    (destination/'case.data').write_bytes(text.encode())
    metadata.update(fingerprint=controls['fingerprint'], native_binary_sha256=sha(destination/'case.data.bin'))
    save_json(destination/'case.data.json', metadata)
    require(perturb or controls['fingerprint'] == controls['original_fingerprint'], 'Endpoint-only fingerprint changed')
    # Only copied metadata is revised; original input/source/executable are never written.
    provenance.update(end=24., files={name:sha(destination/name) for name in origin['files']},
        diagnostic='one-ULP nonzero hu/hv towards +infinity' if perturb else 'unperturbed cold-prefix repeat',
        original_folder=str(SOURCE), original_input_provenance_sha256=sha(SOURCE/'input_provenance.json'))
    save_json(destination/'input_provenance.json', provenance)
    np.savez(destination/'initial_arrays.npz', original=initial['q'], executed=changed,
             delta=changed-initial['q'], area=initial['area'], xy=initial['xy'], rock=initial['rock'])
    receipt = dict(origin=origin, controls=controls, perturbation=evidence,
        staged_files_sha256=provenance['files'], initial_arrays_sha256=sha(destination/'initial_arrays.npz'))
    save_json(destination/'staging.json', receipt)
    return initial, changed, receipt


def execute(folder, exe, identity):
    from braids_support.sediment.lfm_bend_2d import cold_optimized_comparison as previous
    previous.verify_inputs(folder)
    removed = {key:value for key,value in os.environ.items() if key.upper().startswith(('BRAIDS_', 'OMP_'))}
    env = {key:value for key,value in os.environ.items() if key not in removed}
    settings = dict(OMP_NUM_THREADS='8', OMP_DYNAMIC='FALSE', OMP_WAIT_POLICY='PASSIVE',
                    BRAIDS_PROFILE='1', BRAIDS_PROGRESS_STEPS='20000')
    env.update(settings)
    require(sha(exe) == EXE_SHA == identity['executable_sha256'], 'Locked executable identity differs')
    started = time.perf_counter()
    with (folder/'native.log').open('w', encoding='utf-8') as log:
        result = subprocess.run([str(exe), 'case.data'], cwd=folder, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=300)
    record = dict(return_code=result.returncode, wall_s=time.perf_counter()-started, threads=8,
        executable_sha256=sha(exe), source_sha256=identity['source_sha256'],
        removed_inherited_environment=removed, effective_overrides=settings, native_log_sha256=sha(folder/'native.log'),
        timing_claim=False, timing_note='Diagnostic executions may overlap GPU work; not a performance comparison.')
    save_json(folder/'execution.json', record); result.check_returncode()
    previous.verify_inputs(folder)
    require(record['executable_sha256'] == EXE_SHA, 'Executable changed during execution')
    return record


def read_outputs(folder, initial):
    from braids_support.sediment.lfm_bend_2d import run_case as r
    frames = r.shared.read_frames(r.shared.native_frames_path(folder))
    require(len(frames) == 2 and np.allclose([f['time_s'] for f in frames], [0.,24.], rtol=0, atol=1e-7),
            'Diagnostic must produce exactly outputs 0 and24s')
    state = np.stack([frame['q'] for frame in frames]); nc = len(initial['area'])
    require(state.shape == (2,nc,9) and np.isfinite(state).all(), 'Native output layout differs')
    layers = state[:,:,5:9].reshape(2,nc,2,2).transpose(0,2,3,1)
    bed = initial['rock'][None] + layers.sum(axis=(1,2))/1590.
    return dict(time=np.array([frame['time_s'] for frame in frames]), state=state,
        q=state[:,:,:3].transpose(0,2,1), layers=layers, bed=bed,
        secondary=read_secondary_prefix(folder,state), ledger=np.stack([f['ledger'] for f in frames]),
        steps=np.array([frame['steps'] for frame in frames]))


def compare(left, right, area, xy):
    result = {}
    for key in ('state','q','layers','bed','secondary','ledger'):
        a,b = left[key],right[key]; delta=a-b
        result[key]=dict(bitwise_equal=bool(np.array_equal(a.view(np.uint64),b.view(np.uint64))),
                        maximum_absolute_difference=float(abs(delta).max()))
    dz = left['bed']-right['bed']; k,c = np.unravel_index(np.argmax(abs(dz)), dz.shape)
    result['bed'].update(maximum_area_rms_m=float(np.sqrt((dz*dz)@area/area.sum()).max()),
        worst=dict(frame=int(k),time_s=float(left['time'][k]),cell=int(c),xy_m=xy[c].tolist(),signed_difference_m=float(dz[k,c])),
        corner3113_signed_difference_m=float(dz[-1,3113]) if len(area)>3113 else None)
    result['steps_identical']=bool(np.array_equal(left['steps'],right['steps']))
    result['time_bitwise_equal']=bool(np.array_equal(left['time'].view(np.uint64),right['time'].view(np.uint64)))
    return result


def run(tag=TAG):
    from braids_support.sediment.lfm_bend_2d import fast_cpu_cold_comparison as cold
    from .coupled_reference import verify_parent
    require(tag == TAG, 'This bounded diagnostic uses its declared unique tag')
    root, report = HERE/'runs'/tag, HERE/'reports'/tag
    require(not root.exists() and not report.exists(), 'Preserve earlier evidence; destination already exists')
    exe, identity = cold.resolve_runtime(RUNTIME/'xbraids_candidate.exe', RUNTIME/'build.json')
    require(identity['executable_sha256'] == EXE_SHA, 'Wrong locked runtime')
    original_meta = cold.previous.verify_inputs(SOURCE)
    require((original_meta['morfac'],original_meta['cfl'],original_meta['dt'],original_meta['output']) == (25,.8,.0125,24),
            'Reference controls differ')
    protected = {str(SOURCE/name):value for name,value in original_meta['files'].items()}
    for name in ('input_provenance.json','execution.json','frames.bin','secondary_flow.txt'):
        protected[str(SOURCE/name)] = sha(SOURCE/name)
    root.mkdir(); report.mkdir()
    shutil.copy2(__file__, report/Path(__file__).name)
    save_json(report/'plan.json', dict(tag=tag, endpoint_s=24., source=str(SOURCE),
        runtime=str(exe), executable_sha256=EXE_SHA, controls=original_meta,
        protected_files=protected, source_modified=False, gpu_executed=False, long_accuracy_qualified=False))
    control=root/'control'; initial, original, staged0=stage(control, perturb=False)
    # Read only the first two immutable native frames, retaining their original times.
    all_frames=cold.r.shared.read_frames(cold.r.shared.native_frames_path(SOURCE))
    require(len(all_frames)==121, 'Frozen reference must have121outputs')
    # The general reader above expects two outputs; stage a virtual prefix in memory.
    def prefix():
        frames=all_frames[:2];state=np.stack([f['q'] for f in frames]);nc=len(initial['area'])
        layers=state[:,:,5:9].reshape(2,nc,2,2).transpose(0,2,3,1)
        return dict(time=np.array([f['time_s'] for f in frames]),state=state,q=state[:,:,:3].transpose(0,2,1),
            layers=layers,bed=initial['rock'][None]+layers.sum(axis=(1,2))/1590.,
            ledger=np.stack([f['ledger'] for f in frames]),steps=np.array([f['steps'] for f in frames]))
    reference=prefix()
    # Parse the two signed-secondary records without materializing later text frames.
    reference['secondary']=read_secondary_prefix(SOURCE,reference['state'])
    execution0=execute(control,exe,identity); fields0=read_outputs(control,initial)
    control_comparison=compare(fields0,reference,initial['area'],initial['xy'])
    save_json(report/'control_comparison.json',control_comparison)
    require(all(control_comparison[key]['bitwise_equal'] for key in ('state','q','layers','bed','secondary','ledger'))
            and control_comparison['steps_identical'] and control_comparison['time_bitwise_equal'],
            'Unperturbed control failed bitwise reference-prefix check')
    changed_folder=root/'one_ulp'; _, changed, staged1=stage(changed_folder,perturb=True)
    execution1=execute(changed_folder,exe,identity);fields1=read_outputs(changed_folder,initial)
    require(np.array_equal(fields0['state'][0].view(np.uint64),original.view(np.uint64)), 'Control initial state changed')
    require(np.array_equal(fields1['state'][0].view(np.uint64),changed.view(np.uint64)), 'Executed perturbation differs from staged input')
    result=dict(control_vs_original=control_comparison,
        perturbed_vs_control=compare(fields1,fields0,initial['area'],initial['xy']),
        perturbed_vs_original=compare(fields1,reference,initial['area'],initial['xy']),
        staging=dict(control=staged0,one_ulp=staged1),execution=dict(control=execution0,one_ulp=execution1),
        diagnostic_duration_hydraulic_s=24.,diagnostic_duration_morphological_h=24.*25/3600.,
        long_sensitivity_classified=False,original_5mm_gate_relaxed=False,performance_claim=False,
        source_modified=False,delft_modified=False,driver_sha256=sha(__file__))
    for path,digest in protected.items(): require(sha(path)==digest,'Original reference changed: '+path)
    verify_parent(RUNTIME);result['all_reference_and_runtime_identities_retained']=True
    for label,fields in [('control',fields0),('one_ulp',fields1),('reference',reference)]:
        np.savez(report/(label+'_frames.npz'),**fields)
    result['frame_artifact_sha256']={p.name:sha(p) for p in report.glob('*_frames.npz')}
    save_json(report/'comparison.json',result)
    delta=result['perturbed_vs_control']['bed']
    (report/'REPORT.txt').write_text('COLD CPU ONE-ULP DIAGNOSTIC\n'
        'Control0/24s is bitwise identical to the original frozen CPU prefix.\n'
        f"One-ULP hu/hv: maximum bed difference {delta['maximum_absolute_difference']:.12g} m; area-RMS {delta['maximum_area_rms_m']:.12g} m.\n"
        f"Worst location: {delta['worst']}.\n"
        'Only24 hydraulic seconds /0.1667morphologicalhours. This does not classify the11h divergence, excuse the5mm failure, or establish performance.\n',encoding='utf-8')
    return result


def read_secondary_prefix(folder,state):
    nc=state.shape[1];result=np.full((2,nc),np.nan)
    with (folder/'secondary_flow.txt').open() as stream:
        for line in stream:
            if line.startswith('#') or not line.strip():continue
            fields=line.split();t=float(fields[0])
            if t>24.+1e-7:break
            if int(fields[2])!=0:continue
            k=0 if abs(t)<1e-7 else 1 if abs(t-24.)<1e-7 else None
            if k is not None:result[k,int(fields[1])]=float(fields[6])
    require(np.isfinite(result).all() and np.all(result[0]==0.),'Invalid reference secondary prefix')
    return state[:,:,0]*result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    value=run();print(json.dumps(value['perturbed_vs_control'],indent=2))
