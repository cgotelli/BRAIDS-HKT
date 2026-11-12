"""Validation-only parallel source archive prototype, with unchanged checks.

Two independent per-file loops are transformed in memory. Their statement
bodies, all serial schema/inventory/bridge checks, file formats and hashes stay
unchanged. No file system function or retained release source is patched.
"""
import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import inspect
import os
import time


def transformed(function, phase):
    tree = ast.parse(inspect.getsource(function))
    definition = tree.body[0]
    matches = []
    for index, node in enumerate(definition.body):
        if not isinstance(node, ast.For):continue
        name = ast.unparse(node.target)
        if (phase == 'copy' and name == 'original') or (phase == 'verify' and name == '(relative, digest)'):
            matches.append((index, node))
    assert len(matches) == 1, (function.__name__, phase, len(matches))
    index, loop = matches[0]
    if phase == 'copy':
        assert ast.unparse(loop.body[-1]) == 'sources[relative] = SHA(original)'
        body = loop.body[:-1] + [ast.Return(ast.Tuple(
            elts=[ast.Name('relative', ast.Load()), loop.body[-1].value], ctx=ast.Load()))]
        helper = ast.FunctionDef(name='_archive_copy136', args=ast.arguments(
            posonlyargs=[], args=[ast.arg('original')], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=body, decorator_list=[])
        replacement = ast.parse("sources.update(_archive_parallel136('copy', _archive_copy136, _source_files(origin), folder=folder))").body
    else:
        helper = ast.FunctionDef(name='_archive_verify136', args=ast.arguments(
            posonlyargs=[], args=[ast.arg('item')], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=ast.parse('relative, digest = item').body+loop.body, decorator_list=[])
        replacement = ast.parse("_archive_parallel136('verify', _archive_verify136, receipt['sources'].items())").body
    definition.body[index:index+1] = [helper, *replacement]
    ast.fix_missing_locations(tree)
    local = {}
    exec(compile(tree, '<source_parallel_candidate136>', 'exec'), function.__globals__, local)
    return local[function.__name__]


@contextmanager
def install(transport, workers, *, cancelled=lambda:None):
    from braids_core.sediment import river_runtime_lease
    original_prepare = transport.prepare_sources
    original_verify = transport._verify_sources
    name = '_archive_parallel136'
    assert name not in vars(transport)
    stats = []
    def parallel(phase, operation, values, *, folder=None):
        values = list(values)
        if phase == 'copy':
            # An overlapping destination must fail before starting any writer.
            destinations = {}
            for source in values:
                target = (folder/source.name if source.suffix == '.cu' else
                          folder/'python_snapshot'/source.relative_to(transport.ROOT))
                key = os.path.normcase(os.path.abspath(target)).casefold()
                if key in destinations:
                    raise ValueError('Source archive destination collision: '+str(target))
                destinations[key] = source
        serial = workers == 1 or river_runtime_lease._current.get() is not None
        began = time.perf_counter()
        def one(value):
            cancelled()
            return operation(value)
        try:
            if serial:return [one(value) for value in values]
            executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='archive136')
            try:return list(executor.map(one, values))
            finally:executor.shutdown(wait=True, cancel_futures=True)
        finally:
            stats.append(dict(phase=phase, seconds=time.perf_counter()-began,
                              tasks=len(values), workers=1 if serial else workers))
    setattr(transport, name, parallel)
    transport.prepare_sources = transformed(original_prepare, 'copy')
    transport._verify_sources = transformed(original_verify, 'verify')
    try:yield stats
    finally:
        transport.prepare_sources = original_prepare
        transport._verify_sources = original_verify
        delattr(transport, name)
