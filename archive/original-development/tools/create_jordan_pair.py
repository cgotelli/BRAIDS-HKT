"""Create portable, runnable ambient and turbine-on projects from one project.

Use the BRAIDS bundled Python; this is project creation, not a solver build.
Example: python create_jordan_pair.py River.braids.json new_pair_folder
"""
from pathlib import Path
import argparse,sys,json,os

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project',type=Path);parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    runtime=Path(os.environ['LOCALAPPDATA'])/'Programs/BRAIDS-HKT/runtime'
    sys.path.insert(0,str(runtime))
    from braids_desktop import services
    from braids_desktop.jordan_comparison import make_pair
    project=json.loads(args.project.read_text(encoding='utf8'))
    off,on=make_pair(project)
    args.destination.mkdir(parents=True,exist_ok=False)
    for label,doc in [('Ambient',off),('Turbines',on)]:
        services.save_project_folder(args.destination/label,doc,source_base=args.project.parent)
    print('Created Ambient and Turbines. Open each folder in BRAIDS and run them. Retain the same output times. Use Results > Turbines > Jordan comparison with your explicit wake calibration JSON.')

if __name__=='__main__':main()
