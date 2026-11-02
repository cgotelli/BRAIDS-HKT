"""Installed BRAIDS-HKT entry point."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parent
RUNTIME=ROOT/'runtime'
sys.path.insert(0,str(RUNTIME))
sys.dont_write_bytecode=True
if sys.stderr is None or sys.stdout is None:
    folder=Path(os.environ['LOCALAPPDATA'])/'BRAIDS-HKT/logs';folder.mkdir(parents=True,exist_ok=True)
    log=(folder/'startup.log').open('a',encoding='utf-8',buffering=1)
    if sys.stderr is None:sys.stderr=log
    if sys.stdout is None:sys.stdout=log
from braids_desktop.app import main
if __name__=='__main__':raise SystemExit(main(runtime=RUNTIME))
