"""Capture actual QGIS process status and staged/installed alpha129 checks."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--plugin',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--projects',type=Path,default=ROOT/'validation/_inlet129/release_projects')
    p.add_argument('--worker-python',type=Path,default=Path('C:/Users/STREEM/miniconda3/python.exe'))
    a=p.parse_args();out=a.output.resolve()
    if out.exists():raise FileExistsError(out)
    env=dict(os.environ,QT_QPA_PLATFORM='offscreen',BRAIDS_TEST_INSTALLED_PLUGIN=str(a.plugin.resolve()),
        BRAIDS_TEST_RELEASE_PROJECTS=str(a.projects.resolve()),BRAIDS_TEST_WORKER_PYTHON=str(a.worker_python.resolve()))
    env.pop('BRAIDS_TEST_DEVELOPMENT_RUNTIME',None)
    done=subprocess.run(['C:/Program Files/QGIS 3.40.11/bin/python-qgis-ltr.bat',
        str(ROOT/'tests/support/qgis_delivery129.py'),str(out)],cwd=ROOT,env=env,
        capture_output=True,text=True,timeout=1200)
    out.mkdir(parents=True,exist_ok=True)
    (out/'LAUNCHER.log').write_text(done.stdout+'\n'+done.stderr,encoding='utf-8')
    (out/'LAUNCHER.json').write_text(json.dumps(dict(returncode=done.returncode,
        plugin=str(a.plugin.resolve()),projects=str(a.projects.resolve())),indent=2))
    print(done.stdout,flush=True)
    if done.returncode:
        print(done.stderr)
        raise SystemExit('QGIS qualification returned '+str(done.returncode))
    checks=json.loads((out/'CHECKS.json').read_text())['checks']
    if not checks or not all(checks.values()):raise AssertionError(checks)
    print(len(checks),'QGIS checks passed',flush=True)


if __name__=='__main__':main()
