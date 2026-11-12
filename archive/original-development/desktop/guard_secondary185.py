"""Reject an old solver checkpoint before creating a result directory."""
import json,sys
from stage_secondary185 import ROOT,TARGET,OUT,save,sha
sys.path[:0]=[str(TARGET),str(ROOT)]
from braids_core.river_launcher import verify
from braids_core import river_worker
verify()
target=OUT/'old_checkpoint_must_not_run'
old=ROOT/'desktop/examples/two_meanders184/advanced_on/runs/run_20261027_084649_a62dd04f/checkpoint.npz'
assert old.is_file() and not target.exists()
try:river_worker.run(OUT/'signed_cpu_v6/prepared',target,restart=old,until=5.)
except ValueError as exc:
    assert 'identity' in str(exc).lower() or 'checkpoint' in str(exc).lower(),str(exc)
    assert not target.exists()
    save(OUT/'OLD_CHECKPOINT.json',dict(rejected=True,error=str(exc),runtime=sha(TARGET/'RUNTIME.json'),output_created=False))
    print('Old checkpoint rejected before output creation:',exc)
else:raise AssertionError('Old checkpoint unexpectedly accepted')
