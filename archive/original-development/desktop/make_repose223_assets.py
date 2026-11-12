"""Derive the standalone test/delivery harness; no reference outputs edited."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'desktop/example_collapse222.py').read_text(encoding='utf-8')
s=s.replace('stage_collapse222','stage_repose223').replace('from test_collapse222 import document','from test_repose223 import document\nfrom braids_desktop.river_physics import set_choice\nfrom braids_core.sediment.collapse_setup import ORDINARY')
s=s.replace("desktop/examples/collapse222","desktop/examples/repose223")
s=s.replace("for backend,mf,on in [('cpu',25,False),('cpu',1,True),('cpu',25,True),('gpu',25,True)]:", "for backend,mf,on in [('cpu',25,False),('cpu',25,'equalization'),('cpu',1,True),('cpu',25,True),('gpu',25,True)]:")
s=s.replace("('collapse' if on else 'no_collapse')", "('equalization' if on=='equalization' else 'repose' if on else 'no_collapse')")
s=s.replace("            d['name']=", "            if on!='equalization' and not on:\n                panel=SettingsPanel();panel.load_document(d);panel.physics.avalanche_on.setChecked(False);d=panel.document(d);panel.close()\n            if on=='equalization':\n                panel=SettingsPanel();panel.load_document(d);set_choice(panel.physics.avalanche,ORDINARY);panel.physics.collapse_slope.setValue(float(np.tan(np.deg2rad(20.))));d=panel.document(d);panel.close()\n            d['name']=")
(root/'desktop/example_repose223.py').write_text(s,encoding='utf-8')

s=(root/'desktop/report_collapse222.py').read_text(encoding='utf-8')
s=s.replace('stage_collapse222','stage_repose223').replace("root=OUT/'qualification2'","root=OUT/'qualification_final'")
s=s.replace("(OUT/'qualification3').glob", "root.glob").replace("(root,OUT/'qualification3')", "(root,)")
s=s.replace("/avalan.f90'", "/bott3d.f90'")
s=s.replace("desktop/examples/collapse222","desktop/examples/repose223").replace('cpu_MF25_collapse','cpu_MF25_repose').replace('gpu_MF25_collapse','gpu_MF25_repose')
s=s.replace("names=['cpu_MF25_no_collapse','cpu_MF25_repose']", "names=['cpu_MF25_equalization','cpu_MF25_repose']")
s=s.replace("MPM / MPM+GP / VR1993 (CPU and GPU)","MPM + Garcia-Parker (CPU and GPU)")
(root/'desktop/report_repose223.py').write_text(s,encoding='utf-8')
print('Example and figure scripts created')
