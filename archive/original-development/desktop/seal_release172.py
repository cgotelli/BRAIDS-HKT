"""Authenticate the delivered runtime and seal the bounded release evidence."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'desktop/runtime'), str(ROOT), str(ROOT/'desktop')]
from stage_wall172 import sha, save
from braids_desktop.results import ResultsStore
from braids_core import river_worker
import numpy as np


def main():
    base = ROOT/'validation/_delivery172'
    out = ROOT/'validation/sediment/river_morphodynamics/reports/desktop_delivery172'
    runtime = ROOT/'desktop/runtime'
    audit = json.loads((out/'AUDIT.json').read_text())
    receipt = json.loads((ROOT/'desktop/evidence/wall172/DELIVERY.json').read_text())
    assert sha(runtime/'RUNTIME.json') == receipt['installed_runtime'] == audit['stage']['runtime_sha256']
    for name, digest in json.loads((runtime/'RUNTIME.json').read_text())['files'].items():
        assert sha(runtime/name) == digest, name
    for name, digest in audit['source_reference'].items():
        assert sha(ROOT/name) == digest, name
    checks = {}
    for name, count in (('REGRESSIONS.xml',376),('PACKAGE.xml',141),('INSTALLED_UI.xml',28),('INSTALLED.xml',14)):
        suites = ET.parse(base/name).getroot().findall('testsuite')
        assert sum(int(s.attrib['tests']) for s in suites) == count
        assert all(int(s.attrib[k]) == 0 for s in suites for k in ('failures','errors','skipped'))
        checks[name] = dict(passed=count, sha256=sha(base/name))
    installed = {}
    for folder in sorted((base/'installed').iterdir()):
        if not (folder/'CHECK.json').exists():continue
        check = json.loads((folder/'CHECK.json').read_text())
        assert check['comparison']['exact'] and all(check['restart']['exact'].values())
        assert check['runtime_sha256'] == receipt['installed_runtime']
        installed[folder.name] = dict(check=check, files={str(p.relative_to(folder)):sha(p)
            for p in folder.rglob('*') if p.is_file() and p.suffix in ('.json','.npz')})
    assert len(installed) == 12
    # The final installed import, not the candidate, must reject prior identities.
    identity_checks = {}
    for backend in ('cpu','gpu'):
        old = ROOT/f'validation/_wall171/v1/reflection_17_{backend}'
        new = base/f'installed/v1_reflection_17_{backend}'
        try:river_worker.load_prepared(old/'prepared')
        except ValueError as e:prepared_error=str(e)
        else:raise AssertionError('Installed runtime accepted old preparation')
        assert 'Runtime changed' in prepared_error
        project, r = river_worker.load_prepared(new/'prepared')
        expected = river_worker.checkpoint_identity(project,r)
        try:river_worker.read_checkpoint(old/'cancelled/checkpoint.npz',expected)
        except ValueError as e:checkpoint_error=str(e)
        else:raise AssertionError('Installed runtime accepted old checkpoint')
        identity_checks[backend] = dict(preparation=prepared_error,checkpoint=checkpoint_error)
    # Post-promotion replay demonstrates the frozen reference remains usable.
    source = ResultsStore(base/'isolated/groin_wang_fredsoe_cpu/full')
    repeat = ResultsStore(base/'retained_reference/groin_wang_fredsoe_cpu/full')
    assert np.array_equal(source.times,repeat.times)
    for j in range(len(source.times)):
        a,b=source.read(j),repeat.read(j)
        for key in ('q','z','mass','water','bed_boundary'):
            assert np.array_equal(a.arrays[key],b.arrays[key]),(j,key)
    save(out/'INSTALLED.json',dict(controls=installed,identity_checks=identity_checks,
        retained_reference_replayed_exactly=True, reference_harness_lookup_relocated_after_install=True,
        note='Original prototype source is unchanged. Reference CPU lookup now authenticates its retained binary instead of reading the replaced installed table. Initial lookup failed before creating a run; corrected replay is exact.'))
    sources = list((ROOT/'desktop').glob('*172.py')) + [
        ROOT/'tests/test_desktop_delivery172.py',ROOT/'tests/test_desktop_installed172.py',
        ROOT/'braids_core/sediment/reflecting_wall.py',
        ROOT/'validation/sediment/river_morphodynamics/face_activity.py',
        ROOT/'validation/RIVER_MORPHODYNAMICS_STATUS.md',ROOT/'desktop/README.md',
        ROOT/'desktop/PROVENANCE.json',ROOT/'braids_desktop/__init__.py']
    manifest = dict(scope='Canonical exact reflecting-wall pressure, local Desktop preview27 delivery',
        version='0.1.0-preview27',installed=True,qualified=True,groin_jet_agreement_closed=False,
        Delft3D_modified=False,QGIS_modified=False,physical_sediment_laws_changed=False,
        runtime_manifest_sha256=receipt['installed_runtime'],previous_runtime_sha256=receipt['previous_runtime'],
        tests=checks,preinstall_checks=517,postinstall_checks=42,
        packaged_full_controls=16,boundary_checks=32,installed_full_controls=12,
        cancel_resume_exact_pairs=28,retained_reference_replay_exact=True,
        sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='MANIFEST.json'},
        delivery_receipt_sha256=sha(ROOT/'desktop/evidence/wall172/DELIVERY.json'),
        figures_visually_checked=['BOUNDARIES.png','BEDLOAD_PACKAGE.png','JET_SECTIONS.png'],
        next_milestone='Trace and correct the excess turning and wall-side jet displacement on short same-state hydraulic comparisons; no sediment tuning.')
    save(out/'MANIFEST.json',manifest)
    print(json.dumps({k:manifest[k] for k in ('version','installed','preinstall_checks','postinstall_checks','installed_full_controls','groin_jet_agreement_closed')},indent=2))


if __name__ == '__main__':
    main()
