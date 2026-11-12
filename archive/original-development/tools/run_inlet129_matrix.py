"""Sequential natural-river qualification; no concurrent timing workloads."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCES={
    '2m':Path('C:/Users/STREEM/Desktop/aaa/runs/prepare_20261022_204320_bba003.project.json'),
    '5m':Path('C:/Users/STREEM/Desktop/aaa/runs/prepare_20261022_204710_d0ea8c.project.json'),
}

def main():
    p=argparse.ArgumentParser();p.add_argument('--size',choices=SOURCES,default='2m')
    p.add_argument('--backends',nargs='+',default=['cpu','gpu']);p.add_argument('--methods',nargs='+',default=['critical','normal','stage','depth'])
    p.add_argument('--cfl',type=float,default=.5);p.add_argument('--until',type=float,default=100.)
    p.add_argument('--log',type=Path,help='Capture complete child stdout/stderr without PowerShell native-stderr redirection')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    for method in a.methods:
        for backend in a.backends:
            out=a.output/f'{a.size}_{method}_{backend}'
            command=[sys.executable,str(ROOT/'tools/qualify_inlet129.py'),'--source',str(SOURCES[a.size]),
                '--output',str(out),'--backend',backend,'--method',method,'--until',str(a.until),'--cfl',str(a.cfl)]
            if method in ('stage','depth'):command+=['--stage','464.9']
            print('START',out.name,flush=True)
            with (a.log.open('a',encoding='utf-8') if a.log else nullcontext(None)) as stream:
                subprocess.run(command,cwd=ROOT,check=True,stdout=stream,stderr=subprocess.STDOUT if stream else None)
            print('PASS',out.name,flush=True)

if __name__=='__main__':main()
