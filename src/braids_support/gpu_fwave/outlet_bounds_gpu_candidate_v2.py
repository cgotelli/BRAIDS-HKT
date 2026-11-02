"""Schema6: explicit research graph qualification of existing outlet extension.

Preserves schema5's failed-before-evolution evidence. Only the research graph
scope guard accepts the declared extension1; all original remaining graph
constraints and captured-control immutability checks remain enforced.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import json

from . import outlet_bounds_gpu_candidate as v1
from .graph_execution import validate_graph_scope as ORIGINAL_SCOPE

base, HERE = v1.base, v1.HERE
HELPERS = ('outlet_bounds_gpu_candidate_v2.py', 'outlet_bounds_gpu_trace_v2.py')


def graph_scope(data, dt, outlet_extension=0):
    if outlet_extension != 1:
        raise ValueError('Isolated outlet candidate graph requires explicit extension1')
    # All other original validation remains intact. The only expanded research
    # scope is extension1, which the archived kernel already implements.
    ORIGINAL_SCOPE(data, dt, 0)


@contextmanager
def graph_adapters(plan=None):
    from . import coupled_graph
    original_scope, original_graph = coupled_graph.validate_graph_scope, coupled_graph.CoupledTransportGraph
    class OutletGraph(original_graph):
        def __init__(self, solver, **kwargs):
            super().__init__(solver, **kwargs)
            if self.controls[-1] != 1 or solver.reconstruction.outlet_extension != 1:
                raise ValueError('Actual captured outlet control differs')
            self.outlet_evidence = dict(guard_scope_expanded=True, only_expanded_control='outlet_extension0->1',
                actual_captured_outlet_extension=self.controls[-1], original_other_graph_constraints_enforced=True,
                original_control_fingerprint_guard_retained=True, captures=1,
                original_graph_scope_source_sha256=base.sha(HERE/'graph_execution.py'),
                original_graph_source_sha256=base.sha(HERE/'coupled_graph.py'))
            if plan is not None:
                dest=Path(plan['output_path']); diagnostic=dest.with_name(dest.name+'_audit')
                base.write_json(diagnostic/'graph_outlet_control.json', self.outlet_evidence)
        def receipt(self):
            return dict(super().receipt(), outlet_research_scope=self.outlet_evidence)
    coupled_graph.validate_graph_scope, coupled_graph.CoupledTransportGraph = graph_scope, OutletGraph
    try:
        yield
    finally:
        coupled_graph.validate_graph_scope, coupled_graph.CoupledTransportGraph = original_scope, original_graph


class GraphTraces(v1.ExtensionTraces):
    def close(self):
        result=super().close(); path=self.destination/'graph_outlet_control.json'
        if path.exists(): result['artifacts'][path.name]=base.sha(path)
        return result


def prepare(plan_path, output_path, *, until=24., max_steps=None, launch_mode='graph', stop_file=None):
    if launch_mode != 'graph':
        raise ValueError('Schema6 explicitly qualifies graph extension1')
    result=v1.prepare(plan_path, output_path, until=until, max_steps=max_steps, launch_mode=launch_mode, stop_file=stop_file)
    path=Path(plan_path).resolve(); plan=json.loads(path.read_text())
    for name in HELPERS:
        identity=base.sha(HERE/name)
        plan['candidate_source_sha256'][name]=identity
        plan['authenticated_files'][str(HERE/name)]=identity
    plan.update(wrapper_schema=6, wrapper_entrypoint=Path(__file__).name,
                graph_guard_scope_expansion='Only outlet_extension0->1; all other original graph guards retained',
                actual_captured_outlet_extension_required=1)
    base.write_json(path,plan)
    return dict(result, sha256=base.sha(path), wrapper_schema=6)


def run(plan_path, approved_sha256):
    path=Path(plan_path).resolve()
    if base.sha(path)!=approved_sha256: raise ValueError('Reviewed schema6 plan SHA required')
    plan=json.loads(path.read_text())
    if (plan.get('wrapper_schema')!=6 or plan.get('wrapper_entrypoint')!=Path(__file__).name or
        plan['selection']!=v1.selection() or plan['requested_filter_controls']!=v1.CONTROLS or
        plan['effective_filter_controls']!=v1.CONTROLS or plan['launch_mode']!='graph' or
        plan.get('effective_reconstruction_outlet_extension')!=1 or plan.get('requested_input_outlet_extension')!=0 or
        plan.get('actual_captured_outlet_extension_required')!=1 or plan.get('hydraulic_numerical_operator_changed') is not True):
        raise ValueError('Explicit schema6 outlet controls differ')
    for name in v1.HELPERS+HELPERS:
        identity=base.sha(HERE/name)
        if plan['candidate_source_sha256'].get(name)!=identity or plan['authenticated_files'].get(str(HERE/name))!=identity:
            raise ValueError('Unpinned outlet dependency: '+name)
    if v1.stop_path(plan['output_path'],plan['stop_file']).exists(): raise ValueError('Stop signal must be absent')
    with v1.adapters(plan), graph_adapters(plan):
        saved=base.Traces; base.Traces=GraphTraces
        try: return base.run(path,approved_sha256)
        finally: base.Traces=saved


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('prepare');p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--until',type=float,default=24.);p.add_argument('--max-steps',type=int);p.add_argument('--stop-file',type=Path)
    p=subs.add_parser('run');p.add_argument('--plan',type=Path,required=True);p.add_argument('--approved-plan-sha256',required=True)
    args=parser.parse_args()
    result=(prepare(args.plan,args.output,until=args.until,max_steps=args.max_steps,stop_file=args.stop_file)
            if args.command=='prepare' else run(args.plan,args.approved_plan_sha256))
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)
