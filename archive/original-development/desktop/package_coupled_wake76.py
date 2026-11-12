"""Three editable self-contained power-planning projects with saved results."""
import shutil
from pathlib import Path
from coupled_wake76 import ROOT,TARGET,REPORT,read,save,environment
environment()
from braids_desktop import services,project_file

HOME=Path.home()/'Desktop/BRAIDS_Turbine_Coupled76'
def main():
    HOME.mkdir(exist_ok=False);out=[]
    for case,label in [('single','01 Single turbine'),('tandem','02 Two turbines tandem'),('staggered','03 Three turbines staggered')]:
        source=ROOT/'desktop/qualification/coupled_wake76'/(case+'_gpu')
        assert read(source/'DONE.json')['result']['status']=='complete'
        folder=HOME/label;folder.mkdir();root=folder/'Project_data'
        doc=services.save_project_folder(root,read(source/'input.json'),source_base=source);doc['name']=label
        run=root/'runs/coupled_wake';run.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source/'run',run)
        shutil.copytree(source/'prepared',root/'prepared/reference')
        context=read(run/'RUN_CONTEXT.json');context.update(input='../../prepared/reference',input_path_base='run_directory');save(run/'RUN_CONTEXT.json',context)
        session=dict(output='runs/coupled_wake',desktop=dict(model_step='Hydrokinetic turbines',result_step='Turbines',map_styles={'result:velocity_m_s':{'limits':[.6,.86]}}))
        doc=services.save_project_folder(root,doc,session=session);project=folder/'Project.braids.json';project_file.write(project,root,doc)
        launch=folder/'Open project.cmd';launch.write_text('@echo off\ncd /d "'+str(ROOT)+'"\n"'+str(ROOT/'desktop/.venv/Scripts/python.exe')+'" -B "'+str(ROOT/'desktop/launch.py')+'" "'+str(project)+'"\n',encoding='utf-8')
        shutil.copy2(run/'TURBINE_PERFORMANCE.csv',folder/'Turbine performance.csv')
        (folder/'README.txt').write_text('BRAIDS COUPLED TURBINE FLOW - '+label+'\n\nOpen Open project.cmd. Saved 40-second results include real turbine wakes in the conserved hydraulic velocity field. Select Results and Velocity, or Results > Turbines for power/Cp/Ct and CSV export.\n\nThe Estimates only box is OFF. Original OTF force footprint and balanced hydraulic solver are retained. Source-consistent circular rotor moments supply thrust to the flow and mechanical shaft power to the export. Linear zero-depth-mean vertical correction avoids subtracting the resolved wake twice.\n\nThese are fresh startup demonstrations with documented laboratory device data; the three layouts are not identical-device comparisons. Device map and ambient mixing assumptions are saved explicitly. See milestone74 for experimental errors: high-turbulence coupled power errors were about 2% single, 2-5% tandem, and 11-17% staggered. Low-turbulence errors remain larger. The analytical power-planning accuracy must not be claimed for the coupled field.\n\nStatic mesh, straight zero-yaw wakes, up to three rotors and one upstream row. Sediment remains off. The wake-refinement default is 10D and local rotor cells are at most D/2. Two cells across a rotor is a minimum, not an accuracy guarantee.\n',encoding='utf-8')
        out.append(dict(case=case,project=str(project),shortcut=str(launch),source=str(source)))
    (HOME/'README.txt').write_text('Open any numbered folder and double-click Open project.cmd. Each project contains its DEM, settings and saved coupled hydraulic/turbine results. Wakes are in the actual velocity field. Earlier Power75 projects remain unchanged reference-flow-only estimates.\n',encoding='utf-8')
    save(REPORT/'PROJECTS.json',out);print(HOME)
if __name__=='__main__':main()
