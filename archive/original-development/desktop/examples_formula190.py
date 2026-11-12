"""Self-contained short groin projects using the qualified published laws."""
import sys,json
from stage_formula190 import ROOT,OUT
from release_formula190 import RELEASE
sys.path[:0]=[str(RELEASE),str(ROOT)]
from braids_desktop import services
from stage_secondary185 import save

def main():
    rows={}
    for law in ('ancey_recking_2023','recking_2010'):
        original=OUT/'runs/wet'/(law+'_gpu')/'project'
        doc=services.load_project_folder(original)
        root=ROOT/'desktop/examples/bedload190'/law
        assert not root.exists()
        restored=services.save_project_folder(root,doc,source_base=original)
        assert restored['model']['physics']['sediment']['capacity']['method']=='capacity.'+law
        rows[law]=str(root/'braids_project.json')
    save(OUT/'EXAMPLES.json',rows);print(rows)
if __name__=='__main__':main()
