"""Seal the bounded hydraulic diagnostic without claiming a qualified jet fix."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from stage_wall172 import sha,save
from review_hydraulics162 import verify_native


def main():
    out=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_jet173'
    base=ROOT/'validation/_jet173';runtime=ROOT/'desktop/runtime'
    fixed=json.loads((out/'FIXED_BED.json').read_text())
    probes=json.loads((out/'PROBES.json').read_text())
    stencil=json.loads((out/'NATIVE_STENCIL.json').read_text())
    delivery=json.loads((ROOT/'desktop/evidence/wall172/DELIVERY.json').read_text())
    assert sha(runtime/'RUNTIME.json')==delivery['installed_runtime']==fixed['runtime_sha256']==probes['runtime_sha256']
    for name,digest in json.loads((runtime/'RUNTIME.json').read_text())['files'].items():
        assert sha(runtime/name)==digest,name
    for data in (fixed,probes):
        for name,digest in data['inputs'].items():assert sha(ROOT/name)==digest,name
    for data in (fixed,stencil):
        for name,digest in data['native_sources_read_only'].items():assert sha(ROOT/name)==digest,name
    native=verify_native(base/'fixed/delft')
    verify_native(ROOT/'validation/_corner167/native_momentum_fine')
    suites=ET.parse(base/'CHECKS.xml').getroot().findall('testsuite')
    count=sum(int(s.attrib['tests']) for s in suites)
    assert count==185
    assert all(int(s.attrib[k])==0 for s in suites for k in ('failures','errors','skipped'))
    assert len(probes['paired_operator'])==10
    assert fixed['bed_change_BRAIDS']==fixed['bed_change_FLOW']==0
    assert fixed['physical_depth_1mm_pass'] and not fixed['corner_jet_fix_qualified']
    assert not fixed['strict_mixed_q_1e_6_pass']
    assert not fixed['retained40s']['strict_q_1e_9_pass']
    for backend in ('cpu','gpu'):
        check=json.loads((base/f'fixed/{backend}/CHECK.json').read_text())
        assert check['status']['status']=='complete' and check['status']['time_s']==120
    sources=list((ROOT/'desktop').glob('*173.py'))+[
        ROOT/'tests/test_desktop_jet173.py',ROOT/'validation/RIVER_MORPHODYNAMICS_STATUS.md']
    manifest=dict(
        scope='Upstream transient attribution and source-traced groin corner momentum',
        upstream_transient_diagnostic_complete=True,corner_directional_attribution_complete=True,
        native_endpoint_stencil_reconstructed=True,exact_internal_ADI_replay=False,
        corner_jet_fix_qualified=False,production_changed=False,Desktop_runtime_changed=False,
        Delft3D_source_modified=False,Delft3D_runtime_modified=False,QGIS_modified=False,
        runtime_manifest_sha256=sha(runtime/'RUNTIME.json'),
        installed_release='0.1.0-preview27',tests=dict(passed=count,new=30,retained=155,
            failures=0,errors=0,skipped=0,receipt_sha256=sha(base/'CHECKS.xml')),
        paired_operator_states=10,coupled_fixed_bed_runs=3,control_end_s=120,
        strict_mixed_q_backend_1e_6_pass=fixed['strict_mixed_q_1e_6_pass'],
        strict_output_cadence_q_1e_9_pass=fixed['retained40s']['strict_q_1e_9_pass'],
        physical_depth_1mm_pass=fixed['physical_depth_1mm_pass'],native_reference=native,
        sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='MANIFEST.json'},
        figures_visually_checked=['UPSTREAM_ATTRIBUTION.png','TRANSVERSE_TERMS.png',
                                 'FIXED_BED_HISTORY.png','CORNER_TRANSPORT.png'],
        next_milestone='Qualify a conservative leading-corner transport correction if an independent control demonstrates a defect; the jet gate stays open.',
        immediate_next_step='Compare the reconstructed native V stencil and BRAIDS fluxes on a small slip-wall corner control using the same integrated momentum observable; separate projection from transported-trace error.')
    save(out/'MANIFEST.json',manifest)
    print(json.dumps({k:manifest[k] for k in ('upstream_transient_diagnostic_complete',
        'corner_directional_attribution_complete','corner_jet_fix_qualified','production_changed','tests')},indent=2))


if __name__=='__main__':main()
