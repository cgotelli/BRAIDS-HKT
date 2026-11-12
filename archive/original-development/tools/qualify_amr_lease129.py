"""Isolate immutable runtime leasing on untouched installed alpha127 numerics.

No user project/runtime writes; profile_test1_amr creates a fresh private replay.
The only hooks replace file SHA reads and open a bounded lease around the run.
"""
import importlib.util
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path('C:/Users/STREEM/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/braids_qgis/river_runtime')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RUNTIME))
sys.dont_write_bytecode = True


def leased_digest(path):
    from amr_lease129 import digest
    return digest(path)


def main():
    direct='--direct' in sys.argv
    if direct:sys.argv.remove('--direct')
    use_lease='--no-lease' not in sys.argv
    if not use_lease:sys.argv.remove('--no-lease')
    spec = importlib.util.spec_from_file_location('amr_lease129', ROOT/'braids_core/sediment/river_runtime_lease.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from braids_core.sediment import bend_methods_v1, portable_river_runtime, adaptive_river_worker
    from braids_gpu import general_live_transport
    if use_lease:
        for function in (bend_methods_v1.digest, portable_river_runtime.sha, general_live_transport.SHA):
            function.__code__ = leased_digest.__code__
        verify=general_live_transport.verify_sources
        def inventory(receipt):
            origin=general_live_transport.ROOT/receipt.get('kernel_origin','validation/gpu_fwave')
            return frozenset(p.relative_to(general_live_transport.ROOT).as_posix()
                for p in general_live_transport._source_files(origin))
        general_live_transport.verify_sources=lambda folder:module.verified_sources(folder,verify,inventory)
    if direct:
        for name in ('shoreline_solver','current_shoreline_solver'):
            original_module=importlib.import_module('braids_core.sediment.'+name)
            new_spec=importlib.util.spec_from_file_location('braids_core.sediment._qualification129_'+name,
                ROOT/'braids_core/sediment'/f'{name}.py')
            new_module=importlib.util.module_from_spec(new_spec);new_spec.loader.exec_module(new_module)
            if name=='shoreline_solver':original_module.ShorelineSolver.__init__=new_module.ShorelineSolver.__init__
            else:original_module.build_current_shoreline=new_module.build_current_shoreline
        import validation.sediment.river_morphodynamics.face_activity as activity
        new_spec=importlib.util.spec_from_file_location('validation.sediment.river_morphodynamics._qualification129_activity',
            ROOT/'validation/sediment/river_morphodynamics/face_activity.py')
        new_module=importlib.util.module_from_spec(new_spec);new_spec.loader.exec_module(new_module)
        activity.native_hydraulics=new_module.native_hydraulics
        activity.install_hydraulic=new_module.install_hydraulic
        import braids_core.sediment.river_transport as transport
        new_spec=importlib.util.spec_from_file_location('braids_core.sediment._qualification129_transport',
            ROOT/'braids_core/sediment/river_transport.py')
        new_module=importlib.util.module_from_spec(new_spec);new_spec.loader.exec_module(new_module)
        transport.build_transport_river=new_module.build_transport_river
    from braids_core.sediment import bend_native_hydraulics
    native_init=bend_native_hydraulics.NativeHydraulics.__init__;native_count=[0]
    def counted(self,*args,**kwargs):
        native_count[0]+=1
        return native_init(self,*args,**kwargs)
    bend_native_hydraulics.NativeHydraulics.__init__=counted
    original = adaptive_river_worker.run
    record = {}
    def run(*args, **kwargs):
        from contextlib import nullcontext
        with module.runtime_file_lease() if use_lease else nullcontext(None) as lease:
            try:
                return original(*args, **kwargs)
            finally:
                record.update(reads=lease.reads if lease else 0,hits=lease.hits if lease else 0,
                    native_constructions=native_count[0],direct_final_construction=direct,lease=use_lease,
                    scope='Read-only mandatory file leases; identical final installed spatial operators')
    adaptive_river_worker.run = run
    from tools import profile_test1_amr
    out = Path(sys.argv[sys.argv.index('--output')+1])
    try:
        profile_test1_amr.main()
    finally:
        if out.is_dir():
            (out/'LEASE.json').write_text(json.dumps(record,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
