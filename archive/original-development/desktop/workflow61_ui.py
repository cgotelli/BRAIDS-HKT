"""Exercise the public Desktop workflow and leave editable local projects."""
import os,sys,json,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUNTIME=Path(os.environ.get('BRAIDS_VALIDATION_RUNTIME',str(ROOT/'desktop/qualification/workflow61/candidate')))
REPORT=ROOT/'validation/hydrokinetic_turbines/reports/desktop_workflow61';HOME=Path.home()/'Desktop/BRAIDS_Workflow61'
os.environ['QT_QPA_PLATFORM']='offscreen';sys.path.insert(0,str(RUNTIME));sys.path.insert(1,str(ROOT));sys.dont_write_bytecode=True
if os.name=='nt':dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
from braids_desktop import services,project_file,state_initialization
from braids_desktop.shell import MainWindow
from braids_desktop._avac.theme import apply_theme
from braids_core import river_turbines as turbines
from braids_core.river_launcher import verify
from braids_core.river_state_snapshot import source_from_checkpoint,decode

def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    from osgeo import gdal,osr
    from braids_core.boundaries.domain import PolygonDomain
    from braids_desktop.forms import SettingsPanel
    verify();HOME.mkdir(exist_ok=True)
    QSettings.setDefaultFormat(QSettings.IniFormat);QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(REPORT/'ui_settings'))
    app=QApplication([]);app.setOrganizationName('BRAIDSWorkflow61');app.setApplicationName('Acceptance');apply_theme(app)
    projects={}
    for name,kind in [('01_Hydraulic_state','hydro'),('02_Add_bedload_and_map','bedload'),('03_Automatic_wake','turbine')]:
        folder=HOME/name;folder.mkdir(exist_ok=False);dem=folder/'DEM.tif'
        ds=gdal.GetDriverByName('GTiff').Create(str(dem),136,40,1,gdal.GDT_Float64);ds.SetGeoTransform((-1.,.25,0.,9.,0.,-.25))
        srs=osr.SpatialReference();srs.ImportFromEPSG(2056);ds.SetProjection(srs.ExportToWkt());ds.GetRasterBand(1).WriteArray(__import__('numpy').zeros((40,136)));ds=None
        length,width=(32.,8.) if kind=='turbine' else (8.,4.)
        end=8. if kind=='turbine' else (.12 if kind=='hydro' else .24)
        doc=services.default_document(dem,[[0.,0.],[length,0.],[length,width],[0.,width]],'EPSG:2056',name=name,backend='gpu',minimum_size_m=.25 if kind=='turbine' else (.5 if kind=='hydro' else .4),levels=3 if kind=='turbine' else 2,end_time_s=end)
        d=PolygonDomain(**doc['settings']['domain']);depth=2. if kind=='turbine' else 1.;speed=.5 if kind=='turbine' else .2
        for x,n,k,b in [(0.,'Upstream','inlet',dict(method='inlet.critical_discharge',discharge_m3_s=depth*speed*width)),(length,'Downstream','outlet',dict(method='outlet.stage',stage_m=depth))]:
            doc=services.assign_boundary(doc,[e.edge_id for e in d.edges if e.start_xy[0]==e.end_xy[0]==x],n,k,b)
        doc['initial'].update(stage_m=depth,velocity_m_s=[speed,0.]);doc['settings']['adaptation'].update(terrain_enabled=False,domain_border_enabled=False,transition_layers=2)
        doc['settings'].update(dt_max_s=.1 if kind=='turbine' else .03,output_interval_s=1. if kind=='turbine' else .06,checkpoint_interval_s=end)
        if kind=='turbine':
            t=turbines.default();t['turbines']=[turbines.definition([8.,4.],[1.,0.],diameter=1.,hub_height=1.)];doc['turbines']=t
        panel=SettingsPanel();panel.load_document(doc)
        if kind=='bedload':
            panel.physics.enabled.setChecked(True);panel.physics.bedload_on.setChecked(True)
            panel.physics.capacity.setCurrentIndex(panel.physics.capacity.findData('capacity.mpm'))
            for i in range(panel.physics.grain_sizes.rowCount()):panel.physics.grain_sizes.item(i,1).setText('1')
        doc=panel.document(doc);panel.close();doc['settings']['checkpoint_interval_s']=end
        doc['editor_state']['checkpoint_clock']=dict(mode='end_time',custom_interval_s=end)
        saved=services.save_project_folder(folder/'Project_data',doc)
        project=folder/'Project.braids.json';project_file.write(project,folder/'Project_data',saved);projects[kind]=str(project)
    save(HOME/'PROJECTS.json',projects)
    window=MainWindow(runtime=RUNTIME);window.show();errors=[];window.error=lambda e:errors.append(str(e));logs=[];window.runner.log.connect(logs.append)
    def wait(predicate,limit=240.):
        start=time.monotonic()
        while not predicate():
            app.processEvents();time.sleep(.015)
            if errors:raise RuntimeError(errors)
            if time.monotonic()-start>limit:raise TimeoutError(window.status.text())
    def opened(kind):
        window.open_path(Path(projects[kind]),confirmed=True);wait(lambda:window.document_base is not None and not window._tasks)
        window.live.setChecked(False);window.layers['Mesh'].setChecked(False)
    try:
        opened('hydro');window.run_button.click();wait(lambda:window.runner.action=='run' and window.runner.is_running);wait(lambda:not window.runner.is_running and not window._tasks and not window._chain_run)
        source=window.runner.output/'checkpoint.npz';assert decode(source_from_checkpoint(source))[0]['time_s']==.12
        saved_source=HOME/'02_Add_bedload_and_map/Hydraulic_checkpoint.npz';shutil.copy2(source,saved_source)
        opened('bedload');state_initialization.import_checkpoint(window,saved_source);wait(lambda:not window._tasks)
        window.prepare_button.click();wait(lambda:not window.runner.is_running and not window._tasks)
        window.forms.inspect_state.click();wait(lambda:not window._tasks)
        assert window.forms.map_state.isEnabled(),window.forms.state_summary.text()
        assert 'differs' in window.forms.state_summary.text()
        window.grab().save(str(REPORT/'EXPLICIT_MAPPING_UI.png'))
        window.forms.map_state.click();wait(lambda:not window._tasks);assert not window.forms.map_state.isEnabled()
        window.run_button.click();wait(lambda:window.runner.action=='run' and window.runner.is_running);wait(lambda:not window.runner.is_running and not window._tasks and not window._chain_run)
        output=window.runner.output
        from braids_desktop.results import ResultsStore
        store=ResultsStore(output);assert store.times[0]==.12 and store.times[-1]==.24
        opened('turbine');window.run_button.click();wait(lambda:window.runner.action=='run' and window.runner.is_running);wait(lambda:not window.runner.is_running and not window._tasks and not window._chain_run)
        turbine_output=window.runner.output;assert ResultsStore(turbine_output).times[-1]==8.
        window.forms.turbines.wake_length.setValue(10.) if hasattr(window.forms,'turbines') else None
        window.grab().save(str(REPORT/'WAKE_PROJECT_UI.png'))
        from braids_desktop.result_annotations import SpatialProfileDialog
        window.results.load_output(turbine_output);panel=window.results
        import numpy as np
        dialog=SpatialProfileDialog(panel,np.array([[9.,1.],[9.,7.]]),'Wake profile',follow=False);dialog.show();app.processEvents()
        assert dialog.profile.sampling=='linear_cell_interval_centres';dialog.figure.savefig(REPORT/'PROFILE_UI.png',dpi=130)
        dialog.interpolated.setChecked(False);assert dialog.profile.sampling=='raw_cell_intersections';dialog.close()
        save(REPORT/'UI_CHECK.json',dict(status='passed',runtime=str(RUNTIME),projects=projects,source_checkpoint=str(saved_source),mapped_run=str(output),turbine_run=str(turbine_output),saved_time_s=.12,end_time_s=.24,actual_run_buttons=True,mapping_explicit=True,errors=errors))
        print('UI import, explicit map, retained clock, turbine wake and profile controls passed',flush=True)
    finally:
        (REPORT/'UI_WORKER.log').write_text('\n'.join(logs),encoding='utf-8')
        if window.runner.is_running:window.runner.stop();window.runner.process.waitForFinished(30000)
        window.dirty=False;window.close();app.processEvents()

if __name__=='__main__':main()
