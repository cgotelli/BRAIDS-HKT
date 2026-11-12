"""Small source/evidence recovery archive, excluding native binaries and runs."""
import json,zipfile
from stage_supplied220 import ROOT,OUT,REPORT,FILES,sha,save

def main():
    dst=ROOT/'desktop/qualification/supplied220'
    recovery=json.loads((dst/'RECOVERY.json').read_text())
    completion=json.loads((OUT/'COMPLETION.json').read_text())
    names=set(FILES)|set(completion['source_sha256'])
    names.update(r['path'] for r in recovery['restored'])
    names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'desktop').glob('*supplied220.py'))
    names.add('desktop/rebuild_validation220.py')
    names.update(p.relative_to(ROOT).as_posix() for p in REPORT.iterdir() if p.is_file())
    names.update(('desktop/README.md','desktop/PROVENANCE.json','validation/RIVER_MORPHODYNAMICS_STATUS.md',
                 'validation/sediment/river_morphodynamics/README.md',
                 'desktop/qualification/supplied220/RECOVERY.json'))
    sources={n:sha(ROOT/n) for n in sorted(names)}
    archive=dst/'RECOVERED_SOURCE_AND_EVIDENCE.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for n in sources:z.write(ROOT/n,n)
        z.writestr('ARCHIVE_MANIFEST.json',json.dumps(sources,indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        import hashlib
        for n,h in sources.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    save(dst/'ARCHIVE.json',dict(file=archive.name,sha256=sha(archive),members=len(sources),
        scope='Recovered supporting source, new adapter/test source and evidence; no solver binaries or old runs'))
    print('Verified recovery archive:',archive,'members:',len(sources))

if __name__=='__main__':main()
