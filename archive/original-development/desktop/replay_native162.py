"""Restart hydraulic fields through supported FLOW initial files, never source edits."""
import json
import shutil
import sys
from pathlib import Path
import numpy as np
from netCDF4 import Dataset

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'desktop'))
import check_bedload161 as old
import runtime


def main():
    parent=ROOT/'validation/_bedload161_matched/delft_mpm_v8'
    inputs=json.loads((parent/'INPUTS.json').read_text())
    for seconds in (10,30):
        out=ROOT/'validation/_hydraulics162/native_replay'/str(seconds)
        out.mkdir(parents=True,exist_ok=False)
        for name,checksum in inputs['inputs'].items():
            assert old.sha(parent/name)==checksum
            shutil.copy2(parent/name,out/name)
        with Dataset(parent/'trim-sim_P057.nc') as ds:
            j=seconds//5
            stage=np.asarray(ds['S1'][j]);u=np.asarray(ds['U1'][j,0]);v=np.asarray(ds['V1'][j,0])
            # Same field ordering as the official cold-start .ini deck.
            np.savetxt(out/'initial.ini',np.concatenate([stage.T,u.T,v.T]),fmt='%.17e')
        mdf=(out/'sim_P057.mdf').read_text()
        for key,value in dict(Tstop='.01',Dt='.001',Flmap='0 .001 .01',Flhis='0 .001 .01').items():mdf=old.replace(mdf,key,value)
        (out/'sim_P057.mdf').write_text(mdf)
        old.save(out/'INPUTS.json',dict(parent=str(parent),parent_manifest_sha256=old.sha(parent/'INPUTS.json'),
            changes=dict(initial_hydraulic_fields_from_time_s=seconds,end_s=.01,dt_s=.001,
                note='Hydraulic field initialization, not an exact native binary restart; no morphology or turbulence.'),
            Delft3D_source_modified=False,Delft3D_runtime_modified=False,
            inputs={p.name:old.sha(p) for p in out.iterdir() if p.is_file()}))
        print(runtime.run_case(out,configuration='config_flow2d3d.xml',timeout_s=180),flush=True)


if __name__=='__main__':main()
