"""A very shallow fast stream recedes from a wall beyond the vacuum threshold."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import check_wall171 as check
original=check.document


def shallow(folder,backend,angle,case):
    result=original(folder,backend,angle,case)
    from braids_desktop import services
    doc=services.load_project_folder(folder/'project');doc['initial']['stage_m']=.005
    services.save_project_folder(folder/'project',doc)
    return result


if __name__=='__main__':
    check.document=shallow
    check.main()
