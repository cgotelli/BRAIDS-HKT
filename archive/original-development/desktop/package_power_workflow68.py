"""Make a portable editable example with accepted results and a warm state."""
import csv,os,shutil,sys
from pathlib import Path
from power_workflow68 import ROOT,TARGET,REPORT,read,save
sys.path[:0]=[str(TARGET),str(ROOT)]
if os.name=='nt':dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
from braids_desktop import services,project_file

home=Path.home()/'Desktop/BRAIDS_Power68';home.mkdir(exist_ok=False)
source=ROOT/'desktop/qualification/power_workflow68/variable_cpu'
doc=services.save_project_folder(home/'Power_data',read(source/'input.json'),source_base=source)
doc['name']='Turbine power and energy'
output=home/'Power_data/runs/qualified_example';output.parent.mkdir(parents=True,exist_ok=True)
shutil.copytree(source/'run',output)
warm=home/'Power_data/initial_states';warm.mkdir();shutil.copy2(source/'source.npz',warm/'warm_flow_60s.npz')
session=dict(output='runs/qualified_example',desktop=dict(model_step='Hydrokinetic turbines',result_step='Turbines'))
doc=services.save_project_folder(home/'Power_data',doc,session=session)
project=home/'Power.braids.json';project_file.write(project,home/'Power_data',doc)
shutil.copy2(source/'TURBINE_ENERGY.csv',home/'TURBINE_ENERGY.csv')
for row in doc['turbines']['turbines']:
    with (home/(row['id'].replace(' ','_')+'_surface.csv')).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['U_m_s','TSR','Ct','Cp'])
        for table in row['control']['performance']:
            for r in table['curve']:w.writerow([table['inflow_m_s'],r['tip_speed_ratio'],r['thrust_coefficient'],r['power_coefficient']])
(home/'README.txt').write_text('POWER WORKFLOW 68 — LABORATORY FUNCTIONAL EXAMPLE\n\nOpen Power.braids.json in preview62. Results are retained under Power_data/runs/qualified_example. Select Results, then Turbines for histories, energy and CSV.\n\nTo rerun: select a CPU/GPU backend, choose Run > Load useful fields, browse Power_data/initial_states/warm_flow_60s.npz, Prepare, Check saved state, then Run to 80 s. Opening the project intentionally does not select a restart automatically. The original runs and warm state are preserved.\n\nThe local-speed window 0.64–1.1 m/s and 85% electrical efficiency are TEST assumptions, not Mycek manufacturer data. Mechanical Cp/Ct and wake inputs come from the retained Mycek laboratory comparison. Rapid ramps are a functional test, not experimental validation of transient wakes. Do not apply these parameters to a commercial turbine or a Swiss river without appropriate data.\n\nFixed mesh; one upstream wake per exposed rotor. Low-TI 6D power remains overpredicted. No overlapping or turning wake qualification.\n',encoding='utf-8')
shortcut=Path.home()/'Desktop/Open BRAIDS Power68.cmd'
shortcut.write_text('@echo off\r\ncd /d "'+str(ROOT)+'"\r\n"'+str(ROOT/'desktop/.venv/Scripts/python.exe')+'" -B "'+str(ROOT/'desktop/launch.py')+'" "'+str(project)+'"\r\n',encoding='utf-8')
save(REPORT/'EXAMPLE.json',dict(project=str(project),shortcut=str(shortcut),source=str(source),copied_results=True));print(project)
