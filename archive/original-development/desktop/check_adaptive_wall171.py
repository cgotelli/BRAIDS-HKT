"""Repeat the wall wetting/cancellation control through dynamic mesh ownership."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import check_wall171 as check
original=check.document


def adaptive(folder,backend,angle,case):
    result=original(folder,backend,angle,case)
    from braids_desktop import services
    from braids_core.boundaries.domain import PolygonDomain
    doc=services.load_project_folder(folder/'project')
    s=doc['settings'];s['levels']=2
    s['grid']=services.geometry_grid(PolygonDomain(**s['domain']),.5,2)
    s['adaptive']=True;s['adaptation'].update(terrain_enabled=False,buffer_cells=1,interval=.5)
    services.save_project_folder(folder/'project',doc)
    return result


if __name__=='__main__':
    check.document=adaptive
    check.main()
