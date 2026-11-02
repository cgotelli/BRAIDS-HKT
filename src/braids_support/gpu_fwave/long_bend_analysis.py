"""Read-only partial/final audit of the finite GPU bend against two CPU histories.

This is analysis, not a solver or a source of a posteriori acceptance limits.
No reference is modified and no interpolation conceals different output times.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
from .coupled_reference import ROOT
FAST_REFERENCE = ROOT / 'braids_support/sediment/lfm_bend_2d/runs/fast_cpu_cold20h_20261001_v1_mobile_braids'
# Recorded before running this GPU long trajectory. Historical and direct port
# comparisons use separate results, not separate relaxed accuracy thresholds.
LIMITS = dict(bed_rms_m=.003, bed_maximum_m=.005, water_budget_m3=1e-6,
              sediment_budget_kg=1e-5, secondary_budget=1e-8,
              minimum_inventory_kg_m2=-1e-10)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    from braids_gpu.runner import write_json as atomic_write
    atomic_write(path, value)


def serializable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [serializable(v) for v in value]
    return value


def common_times(candidate, reference, tolerance=1e-7):
    """One-to-one exact output-time pairing up to floating summation noise."""
    candidate = np.asarray(candidate, float); reference = np.asarray(reference, float)
    for times in (candidate, reference):
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('Require finite strictly increasing output times')
    left = []; right = []
    for i, time in enumerate(candidate):
        matches = np.flatnonzero(abs(reference - time) <= tolerance)
        if len(matches) > 1: raise ValueError('Ambiguous output-time correspondence')
        if len(matches): left.append(i); right.append(int(matches[0]))
    if len(set(right)) != len(right): raise ValueError('Duplicate reference-time correspondence')
    if not left: raise ValueError('No exact common output times; interpolation is disabled')
    return np.asarray(left, int), np.asarray(right, int)


def weighted_rms(values, area):
    values = np.asarray(values); area = np.asarray(area)
    return np.sqrt(np.sum(values ** 2 * area, axis=-1) / area.sum())


def velocity(q, dry):
    return np.divide(q[:, 1:3], q[:, 0, None], out=np.zeros_like(q[:, 1:3]), where=q[:, 0, None] > dry)


def comparison(gpu, reference, area, dry):
    """Global and local differences; initial-water difference is disclosed."""
    gi, ri = common_times(gpu['time'], reference['time'])
    gq, rq = gpu['q'][gi], reference['q'][ri]
    gz, rz = gpu['bed'][gi], reference['bed'][ri]
    if gq.shape != rq.shape or gz.shape != rz.shape or gz.shape != (len(gi), len(area)):
        raise ValueError('Mismatched state/geometry shape')
    if not all(np.isfinite(v).all() for v in (gq, rq, gz, rz, area)) or np.any(area <= 0):
        raise ValueError('Nonfinite fields or invalid cell area')
    dz = gz - rz; dh = gq[:, 0] - rq[:, 0]
    du = np.linalg.norm(velocity(gq, dry) - velocity(rq, dry), axis=1)
    rms = weighted_rms(dz, area); maximum = np.max(abs(dz), axis=1)
    curves = dict(time_s=gpu['time'][gi], bed_rms_m=rms, bed_maximum_m=maximum,
        depth_rms_m=weighted_rms(dh, area), depth_maximum_m=np.max(abs(dh), axis=1),
        velocity_vector_rms_m_s=weighted_rms(du, area), velocity_vector_maximum_m_s=du.max(axis=1),
        wet_dry_disagreement_cells=np.sum((gq[:, 0] > dry) != (rq[:, 0] > dry), axis=1),
        wet_dry_disagreement_area_m2=((gq[:, 0] > dry) != (rq[:, 0] > dry)) @ area)
    significant = abs(rz - reference['bed'][0]) > .002
    sign_agreement = []
    for k, mask in enumerate(significant):
        sign_agreement.append(float(np.average(np.sign(gz[k, mask] - gpu['bed'][0, mask]) ==
            np.sign(rz[k, mask] - reference['bed'][0, mask]), weights=area[mask])) if mask.any() else None)
    gates = dict(every_common_frame_bed_rms=bool(np.all(rms <= LIMITS['bed_rms_m'])),
                 every_common_frame_bed_maximum=bool(np.all(maximum <= LIMITS['bed_maximum_m'])))
    peak_frame, peak_cell = np.unravel_index(np.argmax(abs(dz)), dz.shape)
    return dict(gates=gates, passed=all(gates.values()), limits=dict(LIMITS),
        compared_frame_count=len(gi), noninitial_frame_count=int(np.sum(gpu['time'][gi] > 1e-7)),
        unmatched_gpu_times_s=np.delete(gpu['time'], gi), temporal_interpolation=False,
        maximum_bed_rms_m=float(rms.max()), maximum_bed_difference_m=float(maximum.max()),
        failed_times_s=gpu['time'][gi[(rms > LIMITS['bed_rms_m']) | (maximum > LIMITS['bed_maximum_m'])]],
        worst_bed_location=dict(cell=int(peak_cell), time_s=float(gpu['time'][gi[peak_frame]]), signed_difference_m=float(dz[peak_frame, peak_cell])),
        initial_bed_maximum_difference_m=float(abs(gpu['bed'][0] - reference['bed'][0]).max()),
        initial_hydraulic_state_maximum_difference=float(abs(gpu['q'][0] - reference['q'][0]).max()),
        erosion_deposition_sign_agreement_where_reference_change_gt_2mm=sign_agreement,
        curves=curves), gi, ri


def budgets(gpu, area, dry):
    """Signed ledgers are inward-positive; preserve that sign in conservation."""
    q, layers, secondary = gpu['q'], gpu['layers'], gpu['secondary']
    if q.ndim != 3 or q.shape[1:] != (3, len(area)) or layers.ndim != 4 or layers.shape[0] != len(q) or layers.shape[-1] != len(area):
        raise ValueError('Invalid GPU inventory shape')
    if secondary.shape != (len(q), len(area)):
        raise ValueError('Invalid GPU secondary inventory shape')
    if not all(np.isfinite(gpu[key]).all() for key in ('q', 'layers', 'bed', 'secondary', 'ledger_water', 'ledger_bed', 'ledger_secondary', 'ledger_source')):
        raise ValueError('Nonfinite GPU state or ledger')
    delta = lambda key: gpu[key] - gpu[key][0]
    water = (q[:, 0] - q[0, 0]) @ area - delta('ledger_water').sum(axis=-1)
    sediment = (layers - layers[0]).sum(axis=1) @ area - delta('ledger_bed').sum(axis=-1)
    signed_secondary = (secondary - secondary[0]) @ area - delta('ledger_secondary').sum(axis=-1) - delta('ledger_source')
    if signed_secondary.shape != (len(q),): raise ValueError('Invalid scalar source ledger shape')
    wet = q[:, 0] > dry
    changed = np.sum(wet != wet[0], axis=1)
    gates = dict(water_conservation=bool(np.max(abs(water)) <= LIMITS['water_budget_m3']),
        sediment_conservation_every_label=bool(np.max(abs(sediment)) <= LIMITS['sediment_budget_kg']),
        sediment_conservation_total=bool(np.max(abs(sediment.sum(axis=1))) <= LIMITS['sediment_budget_kg']),
        signed_secondary_conservation=bool(np.max(abs(signed_secondary)) <= LIMITS['secondary_budget']),
        nonnegative_depth=bool(q[:, 0].min() >= 0),
        nonnegative_sediment=bool(layers.min() >= LIMITS['minimum_inventory_kg_m2']), finite_fields=True)
    return dict(gates=gates, passed=all(gates.values()), limits=dict(LIMITS),
        water_residual_m3=water, sediment_residual_by_label_kg=sediment,
        sediment_total_residual_kg=sediment.sum(axis=1), signed_secondary_residual=signed_secondary,
        maximum_water_residual_m3=float(abs(water).max()), maximum_sediment_label_residual_kg=float(abs(sediment).max()),
        maximum_sediment_total_residual_kg=float(abs(sediment.sum(axis=1)).max()),
        maximum_signed_secondary_residual=float(abs(signed_secondary).max()),
        minimum_depth_m=float(q[:, 0].min()), minimum_inventory_kg_m2=float(layers.min()),
        dry_cells=np.sum(~wet, axis=1), changed_wet_mask_cells=changed,
        wet_dry_transition_exercised=bool(changed.max() > 0),
        wet_dry_qualification_note='A fully wet bend cannot qualify wetting/drying. Saved-frame masks do not observe unsaved intermediate states.')


def load_native(folder):
    from braids_support.sediment.lfm_bend_2d import cold_optimized_comparison as previous
    r = previous.r; folder = Path(folder)
    meta = previous.verify_inputs(folder); execution = read_json(folder / 'execution.json')
    if execution['return_code'] != 0: raise ValueError('CPU reference did not complete successfully')
    frames_path = r.shared.native_frames_path(folder)
    frames = r.shared.read_frames(frames_path)
    time = np.asarray([frame['time_s'] for frame in frames])
    if not np.isclose(time[-1], meta['end'], rtol=0, atol=1e-7): raise ValueError('Incomplete CPU reference')
    state = np.stack([frame['q'] for frame in frames]); ledger = np.stack([frame['ledger'] for frame in frames])
    with np.load(folder / 'analysis_geometry.npz', allow_pickle=False) as source:
        geometry = {key: source[key] for key in source.files}
    layers = state[:, :, 5:9].reshape(len(time), len(geometry['area']), 2, 2).transpose(0, 2, 3, 1)
    bed = geometry['bed'][None] - r.RESERVOIR + layers.sum(axis=(1, 2)) / r.BULK
    return dict(time=time, q=state[:, :, :3].transpose(0, 2, 1), bed=bed, layers=layers,
        metadata=meta, execution=execution, geometry=geometry, folder=folder,
        ledger=ledger, identity=dict(folder=str(folder), frames_sha256=sha(frames_path),
            execution_sha256=sha(folder/'execution.json'), input_provenance_sha256=sha(folder/'input_provenance.json')))


def load_worker(folder):
    from braids_gpu.runner import load_prepared
    folder = Path(folder).resolve(); manifest = read_json(folder / 'run.json')
    if manifest.get('backend') != 'gpu' or manifest.get('automatic_backend_selection') is not False:
        raise ValueError('Not an explicitly selected GPU worker run')
    prepared = Path(manifest['input_folder']); data, identity = load_prepared(prepared)
    if identity != manifest['input_preparation_sha256']: raise ValueError('Prepared input identity changed')
    if not manifest['frames']: raise ValueError('No committed GPU output frames')
    for name, digest in manifest['source_sha256'].items():
        target = (folder / 'source' / name).resolve()
        if target.parent != folder / 'source' or sha(target) != digest: raise ValueError('Frozen GPU runtime identity changed')
    keys = ('q', 'layers', 'secondary', 'bed_elevation', 'ledger_water', 'ledger_bed', 'ledger_secondary', 'ledger_source')
    arrays = {key: [] for key in keys}; times = []
    for record in manifest['frames']:
        path = (folder / record['file']).resolve()
        if path.parent != folder or sha(path) != record['sha256']: raise ValueError('GPU output identity changed')
        with np.load(path, allow_pickle=False) as frame:
            if not np.isclose(frame['time'], record['time_s'], rtol=0, atol=1e-12): raise ValueError('GPU frame time identity changed')
            times.append(float(frame['time']))
            for key in keys: arrays[key].append(frame[key])
    result = {key: np.stack(value) for key, value in arrays.items()}
    result['time'] = np.asarray(times); result['bed'] = result.pop('bed_elevation')
    if result['time'][0] != 0: raise ValueError('Long bend analysis requires the original time-zero output, not a restart fragment')
    common_times(result['time'], result['time'])
    return result, manifest, data, prepared


def morphology(data, geometry):
    from braids_support.sediment.lfm_bend_2d import experiment as e
    dz = data['bed'] - data['bed'][0]; area = geometry['area']
    bend = (geometry['s'] >= e.APPROACH) & (geometry['s'] <= e.BEND_END)
    inner = bend & (geometry['n'] > .75 * e.WIDTH); outer = bend & (geometry['n'] < .25 * e.WIDTH)
    result = dict(erosion_volume_m3=np.maximum(-dz, 0) @ area, deposition_volume_m3=np.maximum(dz, 0) @ area,
        maximum_erosion_m=np.maximum(-dz, 0).max(axis=1), maximum_deposition_m=np.maximum(dz, 0).max(axis=1))
    for name, mask in (('inner_bend_mean_change_m', inner), ('outer_bend_mean_change_m', outer)):
        result[name] = np.average(dz[:, mask], weights=area[mask], axis=1)
    return result


def render(reportdir, gpu, references, comparisons, indices, data, prepared, accounting, morph, audits, *, animation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from .coupled_report import polygons
    cells = polygons(prepared, data); area = data['area']; mf = float(data['morphological_factor'])
    labels = dict(fast_cpu='Fast CPU: same initial water / CFL 0.8', historical='Historical CPU: CFL 0.4')
    last = len(gpu['time']) - 1
    # Select the last output common to both references, not an unmatched partial endpoint.
    shared = set(indices['fast_cpu'][0]) & set(indices['historical'][0]); last = max(shared)
    ref_ids = {key: int(ri[np.flatnonzero(gi == last)[0]]) for key, (gi, ri) in indices.items()}
    change = gpu['bed'] - gpu['bed'][0]
    all_fields = [change] + [ref['bed'] - ref['bed'][0] for ref in references.values()]
    # Shared fixed physical scale over the analyzed interval only, not the future reference.
    cap = max(.001, float(abs(change).max()), *(float(abs(ref['bed'][ri] - ref['bed'][0]).max()) for key, ref in references.items() for ri in [indices[key][1]]))
    def panel(ax, value, title, maximum=cap, unit='Bed change (m)', cmap='RdBu_r', minimum=None):
        collection = PolyCollection(cells, array=value, cmap=cmap, clim=(-maximum if minimum is None else minimum, maximum), edgecolors='none', antialiased=False)
        ax.add_collection(collection); ax.autoscale(); ax.set_aspect('equal'); ax.set(title=title, xlabel='x (m)', ylabel='y (m)')
        ax.figure.colorbar(collection, ax=ax, shrink=.7, label=unit); return collection
    figures = []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), layout='constrained')
    for ax, key in zip(axes[:2], ('historical', 'fast_cpu')):
        ref = references[key]; panel(ax, ref['bed'][ref_ids[key]] - ref['bed'][0], labels[key])
    panel(axes[2], change[last], 'GPU: coupled SSPRK43')
    error = gpu['bed'][last] - references['fast_cpu']['bed'][ref_ids['fast_cpu']]
    panel(axes[3], error*1000, 'GPU - fast CPU (separate scale)', max(1e-6, float(abs(error).max())*1000), 'Bed difference (mm)')
    fig.suptitle(f"Raw cell bed changes at {gpu['time'][last]*mf/3600:.3f} morphological h; MF {mf:g}\nBlue erosion / red deposition; numerical agreement is not experimental validation")
    name='BED_COMPARISON.png'; fig.savefig(reportdir/name, dpi=150); plt.close(fig); figures.append(name)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    for ax, key in zip(axes[0], ('historical', 'fast_cpu', 'gpu')):
        q = gpu['q'][last] if key == 'gpu' else references[key]['q'][ref_ids[key]]
        panel(ax, q[0], 'GPU depth' if key == 'gpu' else labels[key]+' depth', max(float(gpu['q'][last,0].max()), .001), 'Depth (m)', 'Blues', 0)
    for key, record in comparisons.items():
        curve=record['curves']; hours=curve['time_s']*mf/3600
        axes[1,0].plot(hours, curve['bed_rms_m']*1000, label=key+' RMS')
        axes[1,0].plot(hours, curve['bed_maximum_m']*1000, '--', label=key+' maximum')
        axes[1,1].plot(hours, curve['depth_rms_m']*1000, label=key)
        axes[1,2].plot(hours, curve['velocity_vector_rms_m_s'], label=key)
    axes[1,0].axhline(3, color='.4', linestyle=':', label='RMS limit 3 mm'); axes[1,0].axhline(5, color='.6', linestyle=':', label='maximum limit 5 mm')
    for ax, title, unit in zip(axes[1], ('Bed agreement', 'Depth agreement', 'Velocity-vector agreement'), ('mm', 'mm', 'm/s')):
        ax.set(title=title, xlabel='Morphological hours', ylabel='Difference ('+unit+')', ylim=(0,None)); ax.grid(alpha=.2); ax.legend(fontsize=7)
    name='HYDRAULICS_AND_ERRORS.png'; fig.savefig(reportdir/name, dpi=150); plt.close(fig); figures.append(name)
    fig, axes=plt.subplots(2,3,figsize=(14,8),layout='constrained'); hours=gpu['time']*mf/3600
    for key, record in morph.items():
        times=(gpu['time'] if key=='gpu' else references[key]['time'])*mf/3600
        mask=times<=hours[-1]+1e-8
        axes[0,0].plot(times[mask],record['erosion_volume_m3'][mask],label=key+' erosion')
        axes[0,0].plot(times[mask],record['deposition_volume_m3'][mask],'--',label=key+' deposition')
        axes[0,1].plot(times[mask],record['inner_bend_mean_change_m'][mask]*1000,label=key+' inner')
        axes[0,1].plot(times[mask],record['outer_bend_mean_change_m'][mask]*1000,'--',label=key+' outer')
    axes[0,2].plot(hours,accounting['water_residual_m3'],label='Water')
    for g in range(accounting['sediment_residual_by_label_kg'].shape[1]): axes[1,0].plot(hours,accounting['sediment_residual_by_label_kg'][:,g],label=f'Label {g+1}')
    axes[1,0].plot(hours,accounting['sediment_total_residual_kg'],'k--',label='Total')
    axes[1,1].plot(hours,accounting['signed_secondary_residual'],label='Signed hI')
    axes[1,2].plot(hours,accounting['dry_cells'],label='Dry cells'); axes[1,2].plot(hours,accounting['changed_wet_mask_cells'],'--',label='Changed vs initial mask')
    titles=('Redistributed bulk bed volume','Inner deposition / outer erosion trend','Water conservation','Solid mass conservation','Secondary intensity conservation','Observed output wet/dry masks')
    units=('m³','mm','m³ residual','kg residual','signed inventory residual','Cells')
    for ax,title,unit in zip(axes.flat,titles,units):
        ax.set(title=title,xlabel='Morphological hours',ylabel=unit); ax.axhline(0,color='.3',linewidth=.5); ax.grid(alpha=.2); ax.legend(fontsize=7)
    name='EVOLUTION_AND_BUDGETS.png'; fig.savefig(reportdir/name,dpi=150); plt.close(fig); figures.append(name)
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    for key,audit in audits.items():
        for outlet in audit['outlets']:
            rows=outlet['rows']; rings=[row['adjacency_ring'] for row in rows]
            axes[0].plot(rings,[row['cpu_mean_mm'] for row in rows],label=key+f" outlet {outlet['boundary']+1}")
            if key=='fast_cpu': axes[0].plot(rings,[row['gpu_mean_mm'] for row in rows],'k--',label='GPU')
    keys=['GPU','fast_cpu','historical']; values=[audits['fast_cpu']['highpass_gpu_rms_mm'],audits['fast_cpu']['highpass_cpu_rms_mm'],audits['historical']['highpass_cpu_rms_mm']]
    axes[1].bar(keys,values,color=['#287fa2','#bb6942','#808080'])
    axes[0].set(title='Outlet-adjacent mean bed change',xlabel='Adjacency ring inward (0 = boundary owners)',ylabel='Bed change (mm)'); axes[0].axhline(0,color='.3',linewidth=.5); axes[0].legend(fontsize=8)
    axes[1].set(title='Interior four-neighbour high-pass RMS',ylabel='Cell-scale bed change (mm)')
    fig.suptitle('Outlet and cell-scale diagnostics: shared behavior is not proof of physical correctness')
    name='OUTLET_AND_GRIDCHECK.png'; fig.savefig(reportdir/name,dpi=150); plt.close(fig); figures.append(name)
    if animation and len(shared)>1:
        from matplotlib.animation import FuncAnimation, PillowWriter
        chosen=np.array(sorted(shared)); chosen=chosen[np.unique(np.rint(np.linspace(0,len(chosen)-1,min(81,len(chosen)))).astype(int))]
        fig,axes=plt.subplots(1,3,figsize=(12,4.3),layout='constrained'); collections=[]
        for ax,key in zip(axes,('historical','fast_cpu','gpu')):
            collections.append(panel(ax,np.zeros(len(area)), 'GPU' if key=='gpu' else labels[key]))
        title=fig.suptitle('')
        def update(k):
            for collection,key in zip(collections,('historical','fast_cpu','gpu')):
                if key=='gpu': field=change[k]
                else:
                    gi,ri=indices[key]; j=int(ri[np.flatnonzero(gi==k)[0]]); field=references[key]['bed'][j]-references[key]['bed'][0]
                collection.set_array(field)
            title.set_text(f"MF {mf:g}: {gpu['time'][k]*mf/3600:.3f} morphological h (t={gpu['time'][k]:g} s)\nOriginal mobile bed after CPU water spin-up; no bed smoothing")
            return collections+[title]
        movie=FuncAnimation(fig,update,frames=chosen,interval=180,blit=False)
        name='EVOLUTION.gif'; movie.save(reportdir/name,writer=PillowWriter(fps=6),dpi=90); plt.close(fig); figures.append(name)
    return figures


def run(gpu_output, reportdir, *, animation=True):
    """Analyze the immutable committed frames visible at this invocation only."""
    from braids_support.sediment.lfm_bend_2d import cold_optimized_comparison as previous
    from .coupled_report import spatial_audit
    gpu_output=Path(gpu_output).resolve(); reportdir=Path(reportdir).resolve()
    gpu, manifest, data, prepared=load_worker(gpu_output)
    reportdir.mkdir(parents=True,exist_ok=True)
    if (reportdir/'comparison.json').exists() and read_json(reportdir/'comparison.json')['gpu_output']!=str(gpu_output):
        raise ValueError('Report directory belongs to a different run')
    references=dict(fast_cpu=load_native(FAST_REFERENCE), historical=load_native(previous.REFERENCE))
    area=data['area']; dry=float(data['dry_tolerance']); comparisons={}; indices={}; audits={}
    for key,ref in references.items():
        np.testing.assert_allclose(ref['geometry']['xy'],data['xy'].T,rtol=0,atol=1e-10,err_msg='Reference cell ordering changed')
        np.testing.assert_allclose(ref['geometry']['area'],area,rtol=0,atol=1e-12)
        if not np.isclose(ref['metadata']['morfac'],manifest['morfac']): raise ValueError('Mismatched morphological factor')
        comparisons[key],gi,ri=comparison(gpu,ref,area,dry); indices[key]=(gi,ri)
        audits[key],_=spatial_audit(dict(bed=gpu['bed'][gi],native_bed=ref['bed'][ri]),data)
    fast=references['fast_cpu']
    receipt=read_json(prepared/'preparation_receipt.json')
    # The exact text + binary deck provenance verifies forcing, materials and
    # controls, not merely a coincidentally matching initial state.
    origin_files=receipt['origin']['files']
    native_input_match=all(origin_files.get(name)==fast['metadata']['files'][name]
                           for name in ('case.data','case.data.bin'))
    control_match=bool(np.isclose(manifest['cfl'],fast['metadata']['cfl'],rtol=0,atol=1e-14)
                       and np.isclose(manifest['dtmax_s'],fast['metadata']['dt'],rtol=0,atol=1e-14))
    initial_bed_match=bool(np.max(abs(gpu['bed'][0]-fast['bed'][0]))<=1e-12)
    initial_water_match=bool(np.max(abs(gpu['q'][0]-fast['q'][0]))<=1e-12)
    initial_sediment_match=bool(np.max(abs(gpu['layers'][0]-fast['layers'][0]))<=1e-10)
    accounting=budgets(gpu,area,dry)
    complete=manifest.get('status')=='completed' and np.isclose(gpu['time'][-1],fast['time'][-1],rtol=0,atol=1e-7)
    all_reference_outputs=len(indices['fast_cpu'][0])==len(fast['time']) and len(indices['historical'][0])==len(references['historical']['time'])
    direct_pass=bool(comparisons['fast_cpu']['noninitial_frame_count'] > 0 and comparisons['fast_cpu']['passed'] and accounting['passed'] and initial_bed_match and initial_water_match and initial_sediment_match and native_input_match and control_match)
    historical_pass=bool(comparisons['historical']['noninitial_frame_count'] > 0 and comparisons['historical']['passed'] and accounting['passed'])
    morph={key:morphology(ref,ref['geometry']) for key,ref in references.items()}; morph['gpu']=morphology(gpu,fast['geometry'])
    figures=render(reportdir,gpu,references,comparisons,indices,data,prepared,accounting,morph,audits,animation=animation)
    result=dict(schema=1,gpu_output=str(gpu_output),gpu_manifest_sha256=sha(gpu_output/'run.json'),
        analysis_source_sha256=sha(__file__),gpu_status_at_analysis=manifest.get('status','running'),
        final_analyzed_hydraulic_s=float(gpu['time'][-1]),final_analyzed_morphological_h=float(gpu['time'][-1]*manifest['morfac']/3600),
        output_frames=len(gpu['time']),complete_20h_trajectory=bool(complete),all_reference_outputs_compared=bool(all_reference_outputs),
        direct_gpu_port_passed_on_available_frames=direct_pass,historical_agreement_passed_on_available_frames=historical_pass,
        full_direct_gpu_port_qualified=bool(complete and all_reference_outputs and direct_pass),
        full_historical_agreement_qualified=bool(complete and all_reference_outputs and historical_pass),
        limits=dict(LIMITS),comparisons=comparisons,conservation=accounting,morphology=morph,spatial_audit=audits,
        initialization=dict(description='Original uneroded mobile bed and documented CPU 300-second hydraulic spin-up; not GPU-only cold-water warm-up.',
            original_mobile_bed_matches_fast_cpu=initial_bed_match,initial_water_matches_fast_cpu=initial_water_match,
            initial_sediment_matches_fast_cpu=initial_sediment_match,native_input_identity_matches_fast_cpu=native_input_match,
            cfl_and_dt_ceiling_match_fast_cpu=control_match,hydraulic_seed=fast['metadata'].get('seed'),
            no_evolved_bed_restart=fast['metadata'].get('no_evolved_bed_restart'),gpu_secondary_initial_maximum=float(abs(gpu['secondary'][0]).max())),
        reference_identities={key:ref['identity'] for key,ref in references.items()},
        timing=dict(gpu_worker_wall_s=manifest.get('total_wall_s'),fast_cpu_mobile_wall_s=fast['execution']['wall_s'],
            historical_cpu_mobile_wall_s=references['historical']['execution']['wall_s'],
            note='Historical/current runs, not isolated alternating timings. GPU worker time excludes preparation and CPU hydraulic spin-up; native whole-process timing includes its own initialization/output.'),
        figures=[dict(file=name,sha256=sha(reportdir/name)) for name in figures],
        experimental_accuracy_validated=False,equilibrium_established=False,delft_executed=False,delft_modified=False,
        limitations=['Direct GPU-vs-fast-CPU port agreement is separate from the historical numerical-method comparison.',
            'The fast CPU already failed the 5 mm historical maximum-bed criterion; these thresholds are not relaxed.',
            'A fully wet trajectory cannot establish wet/dry-transition robustness.',
            'Shared outlet deposition or cell-scale content does not prove physical correctness.',
            'Twenty morphological hours is a finite comparison duration, not an assertion of equilibrium.',
            'No experimental accuracy, bank erosion, AMR, suspended-load or braided-river qualification is implied.'])
    write_json(reportdir/'comparison.json',serializable(result))
    text=['LONG COUPLED GPU BEND AUDIT',f"Analyzed {result['final_analyzed_morphological_h']:.6g} morphological h / {gpu['time'][-1]:.9g} hydraulic s, MF {manifest['morfac']:g}.",
        f'Complete 20 h trajectory: {complete}; all reference outputs compared: {all_reference_outputs}.',
        result['initialization']['description'],
        f'Direct GPU port pass on available frames: {direct_pass}; full trajectory qualified: {result["full_direct_gpu_port_qualified"]}.',
        f'Historical agreement pass on available frames: {historical_pass}; full trajectory qualified: {result["full_historical_agreement_qualified"]}.']
    for key,record in comparisons.items():
        text.append(f"{key}: maximum bed RMS {record['maximum_bed_rms_m']*1000:.6g} mm (limit 3 mm); maximum cell bed difference {record['maximum_bed_difference_m']*1000:.6g} mm (limit 5 mm).")
        text.append(f"Compared {record['compared_frame_count']} exactly matching outputs, including {record['noninitial_frame_count']} noninitial outputs; {len(record['unmatched_gpu_times_s'])} GPU outputs unmatched (no interpolation).")
    text.extend([f"GPU maximum water / total sediment budget residual: {accounting['maximum_water_residual_m3']:.6g} m3 / {accounting['maximum_sediment_total_residual_kg']:.6g} kg.",
        f"Wet/dry transition exercised in saved output: {accounting['wet_dry_transition_exercised']}.",'',*result['limitations'],'','Figures:',*figures])
    (reportdir/'REPORT.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return serializable(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-output',type=Path,required=True);parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--no-animation',action='store_true');args=parser.parse_args()
    outcome=run(args.gpu_output,args.report,animation=not args.no_animation)
    print(json.dumps({key:outcome[key] for key in ('final_analyzed_morphological_h','full_direct_gpu_port_qualified','full_historical_agreement_qualified')},indent=2))
