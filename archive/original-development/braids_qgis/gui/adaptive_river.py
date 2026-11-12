"""Semi-implicit combined-river workspace and native mesh epochs."""
from dataclasses import asdict
from datetime import datetime,timezone
import json
import os
from pathlib import Path
from uuid import uuid4
from qgis.PyQt.QtCore import Qt,QProcess,QProcessEnvironment,QTimer,QVariant
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QFormLayout,QLabel,QCheckBox,
    QLineEdit,QComboBox,QPushButton,QPlainTextEdit,QFileDialog,QProgressBar,QSlider,QMessageBox)
from qgis.core import (QgsProject,QgsVectorLayer,QgsField,QgsFeature,QgsGeometry,QgsPointXY,
    QgsFillSymbol,QgsGraduatedSymbolRenderer,QgsRendererRange,QgsCoordinateReferenceSystem)
from braids_qgis.core.adaptive_results import read_frame

def recipe_from_dock(dock):
    """Geometry comes from current edits, not an old prepared domain or DEM."""
    from braids_qgis.core.project import grid_for_domain
    d=dock._domain();assign=dock._assignments(d)
    rows=dock._hydraulic_values();hydro={};sed={}
    for name,b in rows.items():
        mode=b['mode']
        if mode not in ('total_discharge','stage') or b.get('hydraulic_series') is not None:raise ValueError('Combined river currently requires constant total-discharge/stage boundaries; no normal/critical or time-series conversion is implicit.')
        key='discharge_m3_s' if mode=='total_discharge' else 'stage_m'
        hydro[name]={'mode':mode,key:b[key]}
        sed[name]=dict(kind='feed',solid_flux_m2_s=[0.],concentration_kg_m3=[0.]) if mode=='total_discharge' else dict(kind='outflow',concentration_kg_m3=[0.])
    if not hydro:raise ValueError('Assign inlet/outlet boundaries first')
    level=dock.amr_levels.value() if dock.amr_enabled.isChecked() else 1
    profile=dict(schema='braids.combined_adaptive_river.v1',physics_mode='hydrodynamics',
        domain=dict(domain_id=d.domain_id,vertices_xy=d.vertices_xy,spatial_reference=d.crs_wkt),
        assignments=[asdict(a) for a in assign],grid=asdict(grid_for_domain(d,dock._base_mesh_size())),levels=level,
        controls=dict(chezy=45.,diameter=dock.d50_mm.value()/1000,morfac=1.,betac=0.,viscosity=0.,diffusivity=0.,threads=dock.threads.value()),
        columns=dict(fraction_ids=['Sand'],solid_density_kg_m3=[2650.],dry_density_kg_m3=[1590.],diameter_m=[dock.d50_mm.value()/1000],transport_thickness_m=.02,underlayer_thickness_m=.2),
        hydraulics=hydro,sediment_boundaries=sed,bank=dict(bank_theta=0.,collapse=False,collapse_slope=.5),
        suspension=dict(bed_factor=1.,sus_factor=0.,diffusivity=0.,temperature=20.,cref=1600.),
        adaptation=dict(interval=.05,coarsen_delay=2,buffer_cells=2,surface=.005,velocity=.2,bed_detail=.02,bed_change=.002,composition=.05,concentration=.01),
        adaptive=level>1,end_time_s=dock.end_time.value(),dt_max_s=dock.dt_max.value(),output_interval_s=dock.output_interval.value())
    stage=dock._initial_stage_value(required=True)
    return dict(profile=profile,dem_path=dock._source_path(dock.dem_layer.currentLayer(),'a terrain DEM'),
        initial=dict(stage_m=stage,velocity_m_s=[0.,0.],layer_thickness_m=[.02,.2],layer_fractions=[[1.],[1.]],concentration_kg_m3=[0.]))

class AdaptiveRiverDialog(QDialog):
    def __init__(self,dock):
        super().__init__(dock);self.dock=dock;self.setWindowTitle('BRAIDS — combined adaptive river');self.resize(810,800)
        self.prepared=None;self.output=None;self.layer=None;self.job=None;self.cancelled=False;self.last_frame=None
        self.runtime=Path(__file__).resolve().parents[1]/'adaptive_runtime'
        layout=QVBoxLayout(self)
        note=QLabel('CPU or GPU • shared semi-implicit f-wave midpoint • common-clock AMR • MF1.\n'
            'Unchecked: hydrodynamics only. Checked: select bedload only or the reviewed combined river bundle. '
            'Short wet/dry and integration-qualified; not a long-time braided-river validation.');note.setWordWrap(True);layout.addWidget(note)
        self.enabled=QCheckBox('Enable sediment transport for this workspace');self.enabled.setChecked(False);layout.addWidget(self.enabled)
        self.physics=QComboBox();self.physics.addItem('Single grain — bedload only','bedload');self.physics.addItem('Combined river bundle (review JSON controls)','combined')
        layout.addWidget(self.physics)
        form=QFormLayout();self.backend=QComboBox();self.backend.addItem('CPU / OpenMP','cpu');self.backend.addItem('GPU / CUDA','gpu')
        self.backend.setCurrentIndex(max(0,self.backend.findData(dock.execution_backend.currentData())))
        self.python=QLineEdit(dock.gpu_python.text());self.workspace=QLineEdit(str(Path(dock.workspace.text() or Path.home()/'Documents/BRAIDS')/'adaptive_river'))
        form.addRow('Compute backend',self.backend);form.addRow('External Python (NumPy/GDAL; CuPy for GPU)',self.python);form.addRow('Output workspace',self.workspace);layout.addLayout(form)
        row=QHBoxLayout()
        for label,callback in [('Use current QGIS domain / DEM',self.from_qgis),('Load setup JSON',self.load_setup),('Load prepared folder',self.load_prepared)]:
            b=QPushButton(label);b.clicked.connect(callback);row.addWidget(b)
        layout.addLayout(row)
        setup_note=QLabel('Explicit setup (review before preparing). This is a separate bundle; legacy sediment options are not silently copied.')
        setup_note.setWordWrap(True);layout.addWidget(setup_note)
        self.recipe=QPlainTextEdit();self.recipe.setPlaceholderText('Use current QGIS domain / DEM, or load a saved setup JSON.\nNo sediment computation starts until the checkbox is enabled.');layout.addWidget(self.recipe,1)
        self.recipe.textChanged.connect(self.invalidate)
        row=QHBoxLayout();self.prepare_button=QPushButton('Prepare');self.run_button=QPushButton('Run');self.stop_button=QPushButton('Stop')
        for b,fn in ((self.prepare_button,self.prepare),(self.run_button,self.run),(self.stop_button,self.stop)):b.clicked.connect(fn);row.addWidget(b)
        layout.addLayout(row)
        self.restart=QLineEdit();self.restart.setPlaceholderText('Optional checkpoint.npz (same backend/runtime/input)');form=QFormLayout();form.addRow('Restart',self.restart);layout.addLayout(form)
        self.progress=QProgressBar();self.progress.setRange(0,100);layout.addWidget(self.progress)
        self.status=QLabel('Not prepared — sediment transport is OFF');self.status.setWordWrap(True);layout.addWidget(self.status)
        row=QHBoxLayout();b=QPushButton('Load results');b.clicked.connect(self.load_results);row.addWidget(b)
        self.variable=QComboBox()
        for title,key in [('Water depth','depth_m'),('Flow speed','velocity_m_s'),('Water surface','stage_m'),('Bed elevation','bed_m')]:self.variable.addItem(title,key)
        row.addWidget(self.variable);self.mesh=QCheckBox('Show mesh');row.addWidget(self.mesh)
        self.play=QPushButton('Play');self.play.clicked.connect(self.toggle_animation);row.addWidget(self.play);layout.addLayout(row)
        self.slider=QSlider(Qt.Orientation.Horizontal);self.slider.setRange(0,0);layout.addWidget(self.slider)
        self.frame_label=QLabel('No result frame');layout.addWidget(self.frame_label)
        self.log=QPlainTextEdit();self.log.setReadOnly(True);self.log.setMaximumHeight(105);layout.addWidget(self.log)
        self.process=QProcess(self);self.process.readyReadStandardOutput.connect(self.logs);self.process.readyReadStandardError.connect(self.logs)
        self.process.finished.connect(self.finished);self.process.errorOccurred.connect(self.process_error)
        self.timer=QTimer(self);self.timer.setInterval(500);self.timer.timeout.connect(self.poll)
        self.animation=QTimer(self);self.animation.setInterval(250);self.animation.timeout.connect(lambda:self.slider.setValue((self.slider.value()+1)%(self.slider.maximum()+1)))
        self.slider.valueChanged.connect(self.show_frame);self.variable.currentIndexChanged.connect(self.show_frame);self.mesh.toggled.connect(self.show_frame)
        self.enabled.toggled.connect(self.invalidate);self.physics.currentIndexChanged.connect(self.invalidate);self.controls()

    def controls(self):
        busy=self.job is not None
        self.prepare_button.setEnabled(not busy)
        self.run_button.setEnabled(self.prepared is not None and not busy)
        self.stop_button.setEnabled(busy and not self.cancelled)
        for w in (self.recipe,self.backend,self.python,self.workspace,self.enabled,self.restart):w.setEnabled(not busy)
        self.physics.setEnabled(not busy and self.enabled.isChecked())

    def error(self,exc):self.status.setText(str(exc));self.log.appendPlainText(str(exc))
    def invalidate(self,*_):self.prepared=None;self.controls()
    def from_qgis(self):
        try:self.recipe.setPlainText(json.dumps(recipe_from_dock(self.dock),indent=2))
        except Exception as exc:self.error(exc)
    def load_setup(self):
        name,_=QFileDialog.getOpenFileName(self,'Combined river setup','','JSON (*.json)')
        if name:
            try:self.recipe.setPlainText(json.dumps(json.loads(Path(name).read_text()),indent=2))
            except Exception as exc:self.error(exc)
    def load_prepared(self):
        name=QFileDialog.getExistingDirectory(self,'Prepared river folder (PREPARED.json)')
        if name:
            if not (Path(name)/'PREPARED.json').is_file():return self.error('PREPARED.json is missing')
            profile=json.loads((Path(name)/'profile.json').read_text());mode=profile.get('physics_mode','combined')
            self.enabled.setChecked(mode!='hydrodynamics');self.physics.setCurrentIndex(max(0,self.physics.findData(mode)))
            self.prepared=Path(name);self.status.setText('Prepared inputs loaded: '+mode);self.controls()
    def new_folder(self,prefix):
        root=Path(self.workspace.text()).expanduser().resolve()
        target=root/(prefix+'_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:6])
        if os.name=='nt' and len(str(target))>130:
            raise ValueError('Output path is too long for the Windows native preparation tools. Choose a shorter workspace, for example Documents/BRAIDS/adaptive_river.')
        root.mkdir(parents=True,exist_ok=True);return target
    def launch(self,action,args,target):
        exe=Path(self.python.text()).expanduser().resolve();script=self.runtime/'braids_core/sediment/adaptive_river_launcher.py'
        if not exe.is_file() or not script.is_file():raise ValueError('External Python or packaged adaptive runtime is missing')
        self.target=target;self.stop_file=target.parent/(target.name+'.stop');self.cancelled=False;self.job=action;self.last_frame=None
        env=QProcessEnvironment.systemEnvironment()
        for name in ('PYTHONPATH','PYTHONHOME','QT_PLUGIN_PATH','QGIS_PREFIX_PATH','GDAL_DRIVER_PATH','GDAL_DATA','PROJ_LIB','PROJ_DATA'):env.remove(name)
        env.insert('PYTHONNOUSERSITE','1');env.insert('PYTHONDONTWRITEBYTECODE','1');env.insert('OMP_NUM_THREADS','4')
        env.insert('PATH',str(exe.parent/'Library/bin')+';'+str(exe.parent)+';'+os.environ.get('SystemRoot','C:/Windows')+'/System32')
        self.process.setProcessEnvironment(env);self.process.setWorkingDirectory(str(target.parent));self.process.setProgram(str(exe))
        self.process.setArguments(['-I',str(script),action,*map(str,args),'--stop-file',str(self.stop_file)])
        self.status.setText('Preparing mesh and fields…' if action.startswith('prepare') else 'Running '+self.backend.currentText())
        self.progress.setValue(0);self.controls();self.timer.start();self.process.start()
    def prepare(self):
        try:
            recipe=json.loads(self.recipe.toPlainText());target=self.new_folder('preparation');source=target.with_suffix('.setup.json')
            recipe['profile']['physics_mode']=self.physics.currentData() if self.enabled.isChecked() else 'hydrodynamics'
            source.write_text(json.dumps(recipe,indent=2),encoding='utf-8');self.prepared=None
            self.launch('prepare-gis',['--recipe',source,'--output',target],target)
        except Exception as exc:self.error(exc)
    def run(self):
        try:
            target=self.new_folder('run');args=['--input',self.prepared,'--output',target,'--backend',self.backend.currentData()]
            if self.restart.text().strip():args+=['--restart',self.restart.text().strip()]
            self.output=target;self.launch('run',args,target)
        except Exception as exc:self.error(exc)
    def stop(self):
        if self.job:
            self.stop_file.write_text('stop\n',encoding='ascii');self.cancelled=True;self.status.setText('Stopping at a safe boundary…');self.controls()
    def logs(self):
        data=bytes(self.process.readAllStandardOutput())+bytes(self.process.readAllStandardError())
        if data:self.log.appendPlainText(data.decode('utf-8',errors='replace'))
    def poll(self):
        try:
            p=self.target/'PROGRESS.json'
            if p.is_file():
                r=json.loads(p.read_text());self.progress.setValue(round(r.get('percent',0)))
                if not self.cancelled:self.status.setText(str(r.get('phase','running'))+' — '+str(round(r.get('percent',0)))+'%')
            if self.job=='run' and (self.output/'frames/INDEX.json').is_file():
                doc=json.loads((self.output/'frames/INDEX.json').read_text());key=doc['frames'][-1]['file']
                if key!=self.last_frame:
                    self.last_frame=key;self.slider.setMaximum(len(doc['frames'])-1);self.slider.setValue(len(doc['frames'])-1);self.show_frame()
        except (OSError,ValueError,KeyError):pass # atomic publications may not yet exist
    def finished(self,code,*_):
        if self.job is None:return
        self.poll();self.logs();action=self.job;self.job=None;self.timer.stop()
        if action=='prepare-gis' and code==0 and (self.target/'prepared/PREPARED.json').is_file():self.prepared=self.target/'prepared'
        state='Cancelled' if self.cancelled else 'Completed' if code==0 else 'Failed — see log'
        self.status.setText(state+': '+str(self.target))
        if code==0 and not self.cancelled:self.progress.setValue(100)
        self.controls()
    def process_error(self,error):
        if error==QProcess.ProcessError.FailedToStart:self.error(self.process.errorString());self.finished(-1)
    def load_results(self):
        name=QFileDialog.getExistingDirectory(self,'River results folder')
        if name:
            try:
                self.output=Path(name);doc,*_=read_frame(self.output);self.slider.setMaximum(len(doc['frames'])-1);self.slider.setValue(len(doc['frames'])-1);self.show_frame()
            except Exception as exc:self.error(exc)
    def toggle_animation(self):
        if self.animation.isActive():self.animation.stop();self.play.setText('Play')
        else:self.animation.start();self.play.setText('Pause')
    def show_frame(self,*_):
        if self.output is None:return
        try:
            doc,row,mesh,values=read_frame(self.output,self.slider.value());key=self.variable.currentData();v=values[key]
            if self.layer is None or QgsProject.instance().mapLayer(self.layer.id()) is None:
                self.layer=QgsVectorLayer('Polygon','BRAIDS adaptive results','memory')
                self.layer.dataProvider().addAttributes([QgsField('value',QVariant.Double)]);self.layer.updateFields()
                QgsProject.instance().addMapLayer(self.layer)
            if (self.output/'RUN_CONTEXT.json').is_file():
                self.layer.setCrs(QgsCoordinateReferenceSystem(json.loads((self.output/'RUN_CONTEXT.json').read_text())['crs']))
            elif self.prepared:
                p=json.loads((self.prepared/'profile.json').read_text());crs=QgsCoordinateReferenceSystem(p['domain']['spatial_reference']);self.layer.setCrs(crs)
            provider=self.layer.dataProvider();provider.truncate();features=[]
            for i,value in enumerate(v):
                xy=mesh['vertices'][mesh['offsets'][i]:mesh['offsets'][i+1]]
                f=QgsFeature(self.layer.fields());f.setGeometry(QgsGeometry.fromPolygonXY([[QgsPointXY(float(x),float(y)) for x,y in xy]]));f.setAttributes([float(value)]);features.append(f)
            provider.addFeatures(features);self.layer.updateExtents();lo=0. if key in ('depth_m','velocity_m_s') else float(v.min());hi=max(lo+1e-12,float(v.max()));ranges=[]
            if key=='depth_m':lo=.001;hi=max(.001000000001,hi)
            for j in range(7):
                a=lo+(hi-lo)*j/7;b=lo+(hi-lo)*(j+1)/7
                color=QColor.fromHsvF((.59 if key=='depth_m' else .01 if key=='velocity_m_s' else .12+.5*j/6),.25+.65*j/6,1.-.35*j/6)
                symbol=QgsFillSymbol.createSimple({'color':color.name(),'outline_style':'solid' if self.mesh.isChecked() else 'no','outline_color':'#606060','outline_width':'0.12'})
                ranges.append(QgsRendererRange(a,b,symbol,f'{a:.4g}–{b:.4g}'))
            if key=='depth_m':
                symbol=QgsFillSymbol.createSimple({'color':'0,0,0,0','outline_style':'no'});ranges.insert(0,QgsRendererRange(-1e100,.000999999999,symbol,'Dry / < 1 mm'))
            self.layer.setRenderer(QgsGraduatedSymbolRenderer('value',ranges));self.layer.setName('BRAIDS '+self.variable.currentText()+' — '+self.output.name);self.layer.triggerRepaint()
            self.frame_label.setText(f"t = {row['time_s']:.6g} s · {len(v)} cells · frame {self.slider.value()+1}/{len(doc['frames'])}")
            if self.dock.iface:self.dock.iface.mapCanvas().refresh()
        except Exception as exc:self.error(exc)
    def closeEvent(self,event):
        if self.job:self.stop();event.ignore();return
        self.animation.stop();super().closeEvent(event)
