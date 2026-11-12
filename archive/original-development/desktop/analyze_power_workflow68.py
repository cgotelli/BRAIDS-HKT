"""Energy and on/off acceptance, plus Desktop-rendered surface editor evidence."""
import csv,json,os,sys
from pathlib import Path
from power_workflow68 import ROOT,TARGET,REPORT,read,save,sha
sys.path[:0]=[str(TARGET),str(ROOT)];os.environ['QT_QPA_PLATFORM']='offscreen'
if os.name=='nt':dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from braids_desktop.results import ResultsStore
from braids_core.river_turbine_surface import integrate_energy

def main():
    home=ROOT/'desktop/qualification/power_workflow68';metrics={};frames={};histories={}
    for backend in ['cpu','gpu']:
        folder=home/('variable_'+backend);assert read(folder/'DONE.json')['result']['status']=='complete'
        data=read(folder/'run/TURBINE_PERFORMANCE.json');energy=integrate_energy(data);histories[backend]=data;frames[backend]=ResultsStore(folder/'run').read(-1)
        for name in {r['turbine_id'] for r in data}:
            rows=[r for r in data if r['turbine_id']==name];transitions=[(a['time_s'],b['time_s'],b['active']) for a,b in zip(rows,rows[1:]) if a['active']!=b['active']]
            e=[r for r in energy if r['turbine_id']==name][-1]
            metrics[backend+' '+name]=dict(transitions=transitions,shaft_Wh=e['shaft_energy_Wh'],electrical_Wh=e['electrical_energy_Wh'])
        metrics[backend+'_water_residual_max_m3']=max(abs(r['water_residual_m3']) for r in read(folder/'run/DIAGNOSTICS.json'))
        assert read(home/('restart_'+backend)/'DONE.json')['exact_restart']
    metrics['q_backend_max']=float(np.max(abs(frames['cpu'].q-frames['gpu'].q)))
    metrics['power_backend_max_W']=max(abs(a['useful_power_W']-b['useful_power_W']) for a,b in zip(histories['cpu'],histories['gpu']))
    assert metrics['q_backend_max']<1e-9 and metrics['power_backend_max_W']<1e-8
    assert any(len(metrics['cpu '+n]['transitions'])>=2 for n in ['Upstream rotor','Downstream rotor'])
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for j,name in enumerate(['Upstream rotor','Downstream rotor']):
        rows=[r for r in histories['cpu'] if r['turbine_id']==name];t=[r['time_s'] for r in rows];e=integrate_energy(rows)
        axes[0,j].plot(t,[r['useful_power_W'] for r in rows],label='Shaft');axes[0,j].plot(t,[r['electrical_power_W'] for r in rows],label='Electrical (85% assumption)')
        axes[1,j].plot(t,[r['shaft_energy_Wh'] for r in e],label='Shaft');axes[1,j].plot(t,[r['electrical_energy_Wh'] for r in e],label='Electrical')
        for ax in axes[:,j]:ax.set(xlabel='Time (s)',title=name,ylim=(0,None));ax.grid(alpha=.2);ax.legend(fontsize=8)
        axes[0,j].set_ylabel('Power (W)');axes[1,j].set_ylabel('Energy since 60 s (Wh)')
    fig.suptitle('UI-configured shutdown/reactivation and energy — functional test, not transient experimental validation');fig.savefig(REPORT/'POWER_ENERGY.png',dpi=150);plt.close(fig)
    save(REPORT/'METRICS.json',metrics)
    from PySide6.QtWidgets import QApplication,QScrollArea
    from braids_desktop.turbines import TurbineEditor
    from braids_desktop._avac.theme import apply_theme
    app=QApplication.instance() or QApplication([])
    from PySide6.QtGui import QFontDatabase,QFont
    for name in ['segoeui.ttf','segoeuib.ttf','seguisym.ttf']:QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+name)
    app.setFont(QFont('Segoe UI',10));apply_theme(app);editor=TurbineEditor();editor.load(read(home/'variable_cpu/input.json')['turbines'])
    editor.resize(680,editor.sizeHint().height());editor.show();app.processEvents();editor.grab().save(str(REPORT/'EDITOR.png'));editor.close()
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
