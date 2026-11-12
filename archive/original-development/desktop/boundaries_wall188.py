"""Retained boundary forcing oracles, shorter current-runtime public cases."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['BRAIDS_AUDIT_RUNTIME']=str(ROOT/'validation/_wall188/desktop_candidate')
import check_boundaries159 as retained
original=retained.document


def document(folder,backend,case):
    doc=original(folder,backend,case)
    from braids_desktop import services
    from braids_core.river_open_boundary import resolve
    end=min(doc['settings']['end_time_s'],40. if case=='schedule' else 2.)
    doc['settings'].update(end_time_s=end,checkpoint_interval_s=end)
    doc['boundary_resolution']=resolve(doc['settings']['hydraulics'],doc['boundary_states'],
        dict(Upstream='inlet',Downstream='outlet'),sediment=doc['model']['physics']['sediment'],ng=1,end_time=end)
    services.save_project_folder(folder,doc)
    return doc


if __name__=='__main__':
    retained.document=document;retained.main()
