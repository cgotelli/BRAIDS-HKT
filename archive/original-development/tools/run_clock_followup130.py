"""Selected paired-backend and measured-retry screens; not timing evidence."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"validation/_speed130/clock_followup"


def main():
    OUT.mkdir(exist_ok=False)
    runs=[]
    for name,backend,clock in (("selected_gpu","gpu","combined"),
                               ("measured_retry_cpu","cpu","combined_measured")):
        command=[sys.executable,str(ROOT/"tools/run_speed130_case.py"),
            "--source",str(ROOT/"validation/_inlet129/natural5m_final/5m_critical_cpu/PROJECT.json"),
            "--output",str(OUT/name),"--backend",backend,"--until","60",
            "--cfl-cap","1","--dry","1e-6","--clock",clock,"--outer-dt","1"]
        print("START",name,flush=True)
        with (OUT/(name+".log")).open("x",encoding="utf-8") as log:
            code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
        runs.append(dict(name=name,returncode=code,command=command))
        (OUT/"MATRIX.json").write_text(json.dumps(runs,indent=2),encoding="utf-8")
        print("PASS" if code==0 else "FAIL",name,flush=True)
    if any(r["returncode"] for r in runs):raise SystemExit(1)


if __name__=="__main__":main()
