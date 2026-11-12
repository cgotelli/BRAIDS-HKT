"""Strict archive-performance analysis reusing milestone135 timing primitives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.analyze_preparation135 import analyze as shared_analyze,figure,read,sha,plt


def analyze(root):
    receipt=read(root/'COMPARISON.json');strict_path=root/'STRICT_COMPARISON.json';strict=read(strict_path)
    if receipt.get('schema')!='braids.preparation136.comparison.v1' or not receipt.get('strict_full_field_comparison'):
        raise ValueError('Milestone136 requires a completed strict receipt, not the provisional tolerance screen')
    if receipt['strict_receipt_sha256']!=sha(strict_path) or strict['benchmark_sha256']!=sha(root/'BENCHMARK.json'):
        raise ValueError('Strict comparison receipt identity differs')
    if not strict['all_pass'] or not all(c['passed'] for c in strict['checks']):
        raise ValueError('Strict field/geometry/clock/budget qualification failed')
    result=shared_analyze(root,expected_limit=0.,field_scope='Same-input alpha135/136 archive preparation optimization: strict unchanged published fields, geometry, hydraulic work and budgets.')
    for name,case in result['cases'].items():
        checks=[c for c in strict['checks'] if c['case']==name]
        if sorted((c['index'],c['variant']) for c in checks)!=[(1,'after'),(2,'after'),(3,'before')]:
            raise ValueError('Incomplete strict comparison inventory')
        if not case['fields_byte_identical'] or not case['budgets_byte_identical']:
            raise ValueError('Strict archive qualification must not permit roundoff differences')
        case['strict_comparisons']=checks
    result.update(schema='braids.preparation136.analysis.v1',analysis_sha256=sha(__file__),
        shared_analysis_sha256=sha(ROOT/'tools/analyze_preparation135.py'),strict_comparison=str(strict_path.resolve()),
        strict_comparison_sha256=sha(strict_path))
    result['interpretation'][2]='Both releases use the exact same v20 project. Every published numerical array is byte-identical; other arrays, geometry, full hydraulic diagnostics and budgets are unchanged. Original conservation limits are retained.'
    return result


def accuracy_figure(summary,out):
    cells=[];labels=[]
    for name,case in summary['cases'].items():
        fields=case['independent_fields'];physical=fields['physical_maxima_over_common_frames']
        exact=lambda value:'Exact' if value else 'FAIL'
        labels.append('CPU8' if name.startswith('cpu8') else 'GPU' if name.startswith('gpu') else name)
        cells.append([f"{physical['depth_maximum_abs_m']:.1e}",f"{physical['momentum_maximum_abs_m2_s']:.1e}",
            f"{physical['bed_maximum_abs_m']:.1e}",exact(case['fields_byte_identical']),
            exact(fields['all_common_geometry_exact']),
            exact(all(c['same_hydraulic_diagnostics'] for c in case['strict_comparisons'])),
            exact(case['budgets_byte_identical'])])
    fig,ax=plt.subplots(figsize=(12,3),layout='constrained');ax.axis('off')
    table=ax.table(cellText=cells,rowLabels=labels,
        colLabels=['Depth max\n(m)','Momentum max\n(m²/s)','Bed max\n(m)','All numerical\noutput arrays','Mesh\ngeometry','Hydraulic\nwork/diagnostics','Water/sediment\nbudgets'],
        cellLoc='center',rowLoc='center',loc='center',bbox=[.02,.28,.96,.48])
    table.auto_set_font_size(False);table.set_fontsize(10)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#d3d9dc')
        if row==0:cell.set_facecolor('#e9eef1');cell.set_text_props(weight='bold')
        elif col>=0:cell.set_facecolor('#eef6ef')
    ax.set_title('Strict before/after accuracy: source archive optimization only',fontsize=14,pad=12)
    counts={len(c['independent_fields']['common_times_s']) for c in summary['cases'].values()}
    frames=str(next(iter(counts))) if len(counts)==1 else '/'.join(map(str,sorted(counts)))
    ax.text(.5,.1,f'All {frames} published times and every stored field checked; all three repeat/release comparisons pass per backend.\nZero numerical tolerance. Conservation and impermeable-wall limits unchanged.',
        ha='center',va='center',transform=ax.transAxes,fontsize=11)
    fig.savefig(out/'PREPARATION_ACCURACY.png',dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--benchmark-root',type=Path,default=ROOT/'validation/_preparation136/release_abba')
    parser.add_argument('--output',type=Path,default=ROOT/'validation/_preparation136/analysis')
    args=parser.parse_args();summary=analyze(args.benchmark_root);args.output.mkdir(parents=True,exist_ok=True)
    figure(summary,args.output,release_labels=('Alpha135','Alpha136'),title='Same-input alpha135 / alpha136 preparation and execution')
    accuracy_figure(summary,args.output)
    (args.output/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({name:case['improvement'] for name,case in summary['cases'].items()},indent=2))


if __name__=='__main__':main()
