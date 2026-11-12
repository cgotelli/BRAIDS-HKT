"""Three editable self-contained power-planning projects with saved results."""
import shutil
from pathlib import Path
from check_power_planning75 import ROOT,TARGET,REPORT,read,save
from braids_desktop import services,project_file

HOME=Path.home()/'Desktop/BRAIDS_Turbine_Power75'
def main():
    HOME.mkdir(exist_ok=False);out=[]
    for case,label in [('single','01 Single turbine'),('tandem','02 Two turbines tandem'),('staggered','03 Three turbines staggered')]:
        source=ROOT/'desktop/qualification/power_planning75'/(case+'_cpu')
        assert read(source/'DONE.json')['result']['status']=='complete'
        folder=HOME/label;folder.mkdir();root=folder/'Project_data'
        doc=services.save_project_folder(root,read(source/'input.json'),source_base=source);doc['name']=label
        run=root/'runs/ambient_reference';run.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source/'run',run)
        shutil.copytree(source/'prepared',root/'prepared/reference')
        context=read(run/'RUN_CONTEXT.json');context.update(input='../../prepared/reference',input_path_base='run_directory');save(run/'RUN_CONTEXT.json',context)
        session=dict(output='runs/ambient_reference',desktop=dict(model_step='Hydrokinetic turbines',result_step='Turbines'))
        doc=services.save_project_folder(root,doc,session=session);project=folder/'Project.braids.json';project_file.write(project,root,doc)
        launch=folder/'Open project.cmd';launch.write_text('@echo off\ncd /d "'+str(ROOT)+'"\n"'+str(ROOT/'desktop/.venv/Scripts/python.exe')+'" -B "'+str(ROOT/'desktop/launch.py')+'" "'+str(project)+'"\n',encoding='utf-8')
        shutil.copy2(run/'POWER_PLANNING.csv',folder/'Power estimates.csv')
        (folder/'README.txt').write_text('BRAIDS POWER PLANNING — '+label+'\n\nOpen Open project.cmd. Saved ambient-flow results and turbine power estimates are included.\n\nModel & Run > Hydrokinetic turbines: Power planning is checked. Edit individual rotor geometry, actual device speed/TSR/Ct/Cp surfaces and documented wake growth. Run starts a fresh turbine-free hydraulic case. CPU/GPU selection stays in General.\n\nResults > Turbines: plot power/Cp/Ct, export selected/all turbines or energy. After a new run (or while it runs), press Calculate / refresh power estimates. The hydraulic map is undisturbed river flow, NOT the analytical wake field. No turbine force acts on that map.\n\nThe included 20-second run is a functional reference-flow example using published laboratory device data, not a new experiment. Compare with milestone74 for experimental errors. Uniform vertical ambient profile; quasi-steady straight wakes; linear overlap and finite rectangular-section normalization. Low-turbulence errors can exceed25%; no universal Swiss-river calibration.\n\nShaft power only unless an electrical efficiency is explicitly supplied. Do not infer device performance from diameter alone. The loaded settings and saved result settings are distinct, as for other BRAIDS runs. Re-run after editing geometry or device data.\n',encoding='utf-8')
        out.append(dict(case=case,project=str(project),shortcut=str(launch),source=str(source)))
    (HOME/'README.txt').write_text('Open one of the three numbered folders and double-click Open project.cmd. Each project contains its DEM, parameters, saved ambient run, and separate power-estimate CSV.\n\nThe chosen method is finite-depth Gaussian wake + linear overlap + device speed/TSR curves + circular rotor-area velocity moments. Power planning is separate from the conserved hydraulic solver.\n',encoding='utf-8')
    save(REPORT/'PROJECTS.json',out);print(HOME)
if __name__=='__main__':main()
