"""A dry startup or a supercritical approach to a raised weir, no tuning."""
import sys
import check_boundaries159 as test
original=test.document
DRY='--dry' in sys.argv
def document(folder,backend,case):
    doc=original(folder,backend,'out_weir')
    mode='dry' if DRY else 'jump'
    doc['name']='General weir '+mode+' '+backend
    doc['initial'].update(stage_m=-1. if mode=='dry' else .2,velocity_m_s=[0.,0.] if mode=='dry' else [3.,0.])
    doc['settings']['hydraulics']['Downstream']['crest_m']=.55
    doc['boundary_resolution']=test.resolve(doc['settings']['hydraulics'],doc['boundary_states'],
        dict(Upstream='inlet',Downstream='outlet'),sediment=None,ng=1,end_time=120.)
    test.services.save_project_folder(folder,doc)
    return doc
test.document=document
if DRY:sys.argv.remove('--dry')
# Store the mode separately: main() parses only the established arguments.
if __name__=='__main__':test.main()
