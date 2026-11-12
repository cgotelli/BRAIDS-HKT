"""Extract Mycek Part II mean power/thrust curves from original PDF vectors."""
import csv,json,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from validation.hydrokinetic_turbines.digitize_mycek_2014 import subpaths
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/hydrokinetic_turbines/reports/power_array62';REF=OUT/'references'
COLORS={0:'rgb(0%, 0%, 0%)',2:'rgb(58.006287%, 0%, 83.006287%)',4:'rgb(82.029724%, 40.989685%, 12.010193%)',6:'rgb(50%, 50%, 50%)',8:'rgb(19.993591%, 80.076599%, 19.993591%)',10:'rgb(12.010193%, 56.054688%, 100%)',12:'rgb(100%, 0%, 0%)'}
rows=[];fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for ax,field,page,figure,ty,y0,ymax,limit in zip(axes,['Cp','Ct'],[7,8],[4,5],[454.718994,349.425003],[818.464367,686.988727],[4115.539232,4115.503439],[.6,1.2]):
    file=REF/f'part2_figure{figure}.svg';paths=[e for e in ET.parse(file).iter() if e.tag.rsplit('}',1)[-1]=='path']
    for spacing,colour in COLORS.items():
        found=[]
        for e in paths:
            if e.get('transform')==f'matrix(0.0285599, 0, 0, -0.0285599, 91.800995, {ty})' and e.get('stroke')==colour and e.get('stroke-width')=='2.5':
                c=max(subpaths(e.get('d','')),key=len)
                if len(c)>3:found.append(c)
        assert len(found)==1,(field,spacing,len(found))
        a=np.array(found[0]);x=(a[:,0]-1013.488112)/((6712.998885-1013.488112)/9);v=(a[:,1]-y0)/((ymax-y0)/limit)
        assert np.all(np.diff(x)>0)
        ax.plot(x,v,'o-',ms=3,label='Single' if spacing==0 else f'{spacing}D')
        for xx,vv in zip(x,v):rows.append(dict(field=field,spacing_D=spacing,nominal_TSR=xx,coefficient=vv,inflow_m_s=.8,TI_percent=3,upstream_nominal_TSR=4,pdf_page=page,figure=figure))
    ax.set(xlabel='Downstream TSR based on flume inflow',ylabel=field+' based on flume inflow',title=f'Original Figure {figure}a: mean values, TI=3%');ax.grid(alpha=.2);ax.legend(ncol=2,fontsize=8)
fig.savefig(OUT/'EXPERIMENTAL_POWER_TARGETS.png',dpi=170)
with (REF/'mycek_part2_performance.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
meta=dict(doi='10.1016/j.renene.2013.12.048',pdf_sha256=hashlib.sha256((REF/'mycek_2014_part2.pdf').read_bytes()).hexdigest(),
 extraction='Mean polyline vectors; original linear tick positions. No measured samples or standard-deviation extraction.',
 caution='Zero spacing denotes the isolated rotor, not a tandem. Nominal TSR and coefficients use U_inf, not reconstructed local approach speed.',
 extraction_allowance_coefficient=.001,allowance_type='Plot/axis extraction allowance, not experimental uncertainty',
 selected_points={str(d):{field:float(np.interp(4,[r['nominal_TSR'] for r in rows if r['field']==field and r['spacing_D']==d],[r['coefficient'] for r in rows if r['field']==field and r['spacing_D']==d])) for field in ('Ct','Cp')} for d in (0,6,10)})
(REF/'DIGITIZATION.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta['selected_points'],indent=2))
