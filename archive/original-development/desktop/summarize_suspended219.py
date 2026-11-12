"""Extract S6 numerical evidence without mixing retained S3-S5 cases."""
import json
from stage_suspended219 import OUT,TARGET,REPORT,sha,save

METHODS=('suspension.van_rijn_1984','suspension.zyserman_fredsoe_1994','suspension.camenen_larson_2008')

def main():
    results=OUT/'packaged_final_v2';rows=[];temporal=[]
    for source in results.rglob('project.json'):
        doc=json.loads(source.read_text())
        sediment=doc.get('model',{}).get('physics',{}).get('sediment') or {}
        method=sediment.get('suspension')
        if method not in METHODS:continue
        path=source.parent/'run/DIAGNOSTICS.json'
        if not path.exists():continue
        diag=json.loads(path.read_text())
        rows.append(dict(method=method,case=str(source.parent.relative_to(results)),
            water=max(abs(r['water_residual_m3']) for r in diag),
            sediment=max(abs(x) for r in diag for x in r['class_residual_kg']),
            minimum_stock=min(r['minimum_stock'] for r in diag),
            minimum_suspended=min(r['minimum_suspension'] for r in diag),
            wall_mass=max(abs(r['wall_integrals'][0]) for r in diag),
            final_time=diag[-1]['time_s']))
    for p in results.glob('*/TEMPORAL.json'):
        doc=json.loads((p.parent/'0/project.json').read_text())
        method=doc['model']['physics']['sediment']['suspension']
        if method in METHODS:
            temporal.append(dict(method=method,case=p.parent.name,errors=json.loads(p.read_text())))
    assert len(rows)==126,len(rows)
    assert len(temporal)==6,len(temporal)
    summary={m:dict(cases=sum(r['method']==m for r in rows),
        water=max(r['water'] for r in rows if r['method']==m),
        sediment=max(r['sediment'] for r in rows if r['method']==m),
        minimum_stock=min(r['minimum_stock'] for r in rows if r['method']==m),
        minimum_suspended=min(r['minimum_suspended'] for r in rows if r['method']==m),
        wall_mass=max(r['wall_mass'] for r in rows if r['method']==m)) for m in METHODS}
    value=dict(runtime=sha(TARGET/'RUNTIME.json'),summary=summary,temporal=temporal,cases=rows)
    save(OUT/'QUALIFICATION_METRICS.json',value);save(REPORT/'QUALIFICATION_METRICS.json',value)
    print(json.dumps(dict(summary=summary,temporal=temporal),indent=2))

if __name__=='__main__':main()
