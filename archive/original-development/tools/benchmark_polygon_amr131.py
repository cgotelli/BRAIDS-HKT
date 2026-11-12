"""Bounded exact geometry-construction microbenchmark, not solver throughput."""
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from braids_core.sediment import river_polygon_banks as current
from test_river_polygon_banks import rectangles


def main():
    baseline = ROOT / 'validation/_amr131/source_before/braids_core/sediment/river_polygon_banks.py'
    spec = importlib.util.spec_from_file_location('braids_core.sediment._polygon131_reference', baseline)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    result = {'scope': 'geometry-only CPU preprocessing shared by CPU/GPU; native engine loading excluded',
              'threads': 1, 'repetitions': 4, 'cases': {}}
    for nx, ny in ((24, 25), (100, 100)):
        data, polygons = rectangles(nx, ny)
        current._cell_geometry.cache_clear()
        start = time.perf_counter()
        original = frozen.topology(data, polygons)
        baseline_cold = time.perf_counter() - start
        start = time.perf_counter()
        candidate = current.topology(data, polygons)
        candidate_cold = time.perf_counter() - start
        for key in ('area', 'xy', 'vertex_offsets', 'vertices', 'offset', 'neighbour', 'reverse'):
            np.testing.assert_array_equal(candidate[key], original[key])
        assert candidate['sha256'] == original['sha256']
        samples = {'frozen_active': [], 'cached_active': [], 'cached_inactive': []}
        for index in range(4):
            order = list(samples) if index % 2 == 0 else list(reversed(samples))
            for name in order:
                start = time.perf_counter()
                if name == 'frozen_active': frozen.topology(data, polygons)
                else: current.topology(data, polygons, transfers=name == 'cached_active')
                samples[name].append(time.perf_counter() - start)
        medians = {name: float(np.median(values)) for name, values in samples.items()}
        result['cases'][str(nx * ny)] = dict(cells=nx*ny, faces=len(data['owner']),
            cold_s={'frozen': baseline_cold, 'candidate': candidate_cold}, samples_s=samples,
            medians_s=medians, same_enabled_geometry=True, cache=current._cell_geometry.cache_info()._asdict(),
            cached_active_speedup=medians['frozen_active']/medians['cached_active'],
            cached_inactive_speedup=medians['frozen_active']/medians['cached_inactive'])
    output = ROOT / 'validation/_amr131/POLYGON_MICROBENCHMARK.json'
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
