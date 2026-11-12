"""Fresh public preparation and restart after qualification catalog publication."""
import os,sys,subprocess
from stage_formula190 import ROOT,OUT
from release_formula190 import RELEASE

if '--child' in sys.argv:
    sys.argv.remove('--child')
    import run_formula190 as driver
    driver.TARGET=RELEASE;driver.OUT=OUT/'release_qualification'
    driver.main()
else:
    os.environ['OMP_NUM_THREADS']='4'
    for law in ('ancey_recking_2023','recking_2010'):
        for backend in ('cpu','gpu'):
            path=OUT/'logs'/('release_'+law+'_'+backend+'.log')
            with path.open('w') as stream:
                result=subprocess.run([sys.executable,__file__,'--child','--backend',backend,'--law',law],
                                      cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            print(law,backend,result.returncode,flush=True)
            if result.returncode:raise RuntimeError(str(path))
