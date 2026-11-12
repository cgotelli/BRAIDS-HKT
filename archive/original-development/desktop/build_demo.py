"""Stage the requested local river example without touching its source files."""
import launch
import json
from pathlib import Path
from braids_desktop.demo import create_river_demo
from braids_desktop.workspace import digest

base=Path(__file__).resolve().parent
source=Path('C:/Users/STREEM/Desktop/01010101');root=base/'examples/river_demo'
before=digest(source/'braids_project.json')
document=create_river_demo(root,source=source)
assert before==digest(source/'braids_project.json')
print(json.dumps(dict(demo=str(root),source_unchanged=True,backend=document['model']['request']['execution']['backend']),indent=2))
