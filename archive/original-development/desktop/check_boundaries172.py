"""Short explicit-clock package replay of the retained boundary trace checks."""
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('BRAIDS_AUDIT_RUNTIME',str(ROOT/'validation/_delivery172/candidate'))
sys.path.insert(0,str(ROOT/'desktop'))
import check_boundaries159 as retained
original=retained.document


def short_document(folder,backend,case):
    doc=original(folder,backend,case)
    from braids_desktop import services
    from braids_core.river_open_boundary import resolve
    # Keep the whole 40-second schedule; other checks exercise startup/control,
    # not equilibrium. Input changes are explicit and saved in each project.
    end=min(doc['settings']['end_time_s'],40. if case=='schedule' else 10.)
    doc['settings'].update(end_time_s=end,checkpoint_interval_s=end)
    doc['boundary_resolution']=resolve(doc['settings']['hydraulics'],doc['boundary_states'],
        dict(Upstream='inlet',Downstream='outlet'),sediment=doc['model']['physics']['sediment'],ng=1,end_time=end)
    services.save_project_folder(folder,doc)
    return doc


if __name__=='__main__':
    retained.document=short_document
    retained.main()
