"""Run the four established laws through rebuilt coupled CPU/GPU bindings."""
import os,sys,subprocess
from stage_formula190 import ROOT,TARGET,OUT

if '--child' in sys.argv:
    sys.argv.remove('--child')
    import check_smart183 as driver
    driver.OUT=OUT/'legacy_regression'
    driver.main()
else:
    os.environ['OMP_NUM_THREADS']='4'
    for law in ('mpm','ashida_michiue_generalized','wang_fredsoe','smart_jaggi'):
        for backend in ('cpu','gpu'):
            path=OUT/'logs'/('legacy_'+law+'_'+backend+'.log')
            with path.open('w') as stream:
                result=subprocess.run([sys.executable,__file__,'--child','--runtime',str(TARGET),
                    '--tag','wet','--backend',backend,'--law',law],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            print(law,backend,result.returncode,flush=True)
            if result.returncode:raise RuntimeError(str(path))
