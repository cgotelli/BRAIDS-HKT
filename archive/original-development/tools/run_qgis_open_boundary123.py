"""Capture real QGIS exit status and complete alpha123 qualification evidence."""
from pathlib import Path
import argparse,json,os,subprocess
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plugin',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve()
    if out.exists():raise FileExistsError(out)
    env=dict(os.environ,QT_QPA_PLATFORM='offscreen',BRAIDS_TEST_INSTALLED_PLUGIN=str(a.plugin.resolve()))
    env.pop('BRAIDS_TEST_DEVELOPMENT_RUNTIME',None)
    done=subprocess.run(['C:/Program Files/QGIS 3.40.11/bin/python-qgis-ltr.bat',str(ROOT/'tests/support/qgis_open_boundary123.py'),str(out)],
        cwd=ROOT,env=env,capture_output=True,text=True,timeout=1200)
    out.mkdir(parents=True,exist_ok=True)
    (out/'LAUNCHER.log').write_text(done.stdout+'\n'+done.stderr,encoding='utf-8')
    (out/'LAUNCHER.json').write_text(json.dumps(dict(returncode=done.returncode,plugin=str(a.plugin.resolve())),indent=2))
    print(done.stdout,flush=True)
    if done.returncode:print(done.stderr);raise SystemExit('QGIS launcher returned '+str(done.returncode))
    checks=json.loads((out/'CHECKS.json').read_text())['checks'];assert all(checks.values())
    print(len(checks),'QGIS checks passed',flush=True)
