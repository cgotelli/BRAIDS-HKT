"""Sloping wet/dry lake-at-rest regression on actual cropped Desktop cells."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import check_wall171 as check
original=check.document


def lake(folder,backend,angle,case):
    result=original(folder,backend,angle,case)
    from braids_desktop import services
    doc=services.load_project_folder(folder/'project');doc['initial']['velocity_m_s']=[0.,0.]
    services.save_project_folder(folder/'project',doc)
    return result


if __name__=='__main__':
    check.document=lake
    check.main()
