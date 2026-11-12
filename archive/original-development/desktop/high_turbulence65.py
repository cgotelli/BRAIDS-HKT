"""Independent high-TI rotor-performance control from Mycek's PDF vectors.

Offline measured-profile calculation, not a high-TI shallow-water simulation.
"""
import csv,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from validation.hydrokinetic_turbines.digitize_mycek_2014 import subpaths
from validation.hydrokinetic_turbines.rotor_inflow64 import disk_quadrature,moments
REPORT=ROOT/'validation/hydrokinetic_turbines/reports/coupled_inflow65'
REF=REPORT.parent/'power_array62/references'


def main():
    colours={0:'rgb(0%, 0%, 0%)',6:'rgb(19.993591%, 80.076599%, 19.993591%)',10:'rgb(100%, 0%, 0%)'}
    curves={};records=[]
    for field,figure,ty,y0,ymax,limit in [('Cp',4,454.718994,818.464367,4115.539232,.6),('Ct',5,349.425003,686.988727,4115.503439,1.2)]:
        paths=[e for e in ET.parse(REF/f'part2_figure{figure}.svg').iter() if e.tag.rsplit('}',1)[-1]=='path']
        for spacing,colour in colours.items():
            found=[]
            for e in paths:
                if e.get('transform')==f'matrix(0.0285599, 0, 0, -0.0285599, 314.572998, {ty})' and e.get('stroke')==colour and e.get('stroke-width')=='2.5':
                    c=max(subpaths(e.get('d','')),key=len)
                    if len(c)>3:found.append(c)
            assert len(found)==1,(field,spacing,len(found))
            a=np.array(found[0]);x=(a[:,0]-1013.488112)/((6712.998885-1013.488112)/9);v=(a[:,1]-y0)/((ymax-y0)/limit)
            assert np.all(np.diff(x)>0)
            curves[field,spacing]=(x,v)
            records.extend(dict(field=field,spacing_D=spacing,nominal_TSR=float(xx),coefficient=float(vv),TI_percent=15,inflow_m_s=.83,figure=figure) for xx,vv in zip(x,v))
    with (REPORT/'MYCEK_TI15_TARGETS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    with (ROOT/'validation/hydrokinetic_turbines/measurements/mycek_2014_figure_A12_velocity.csv').open() as f:data=list(csv.DictReader(f))
    y,z,w=disk_quadrature(.5);radius=np.hypot(y,z);out=[]
    for spacing in [6,10]:
        selected=sorted((float(r['y_over_d']),float(r['mean_u_over_u_infinity'])) for r in data if float(r['ambient_turbulence_percent'])==15 and float(r['x_over_d'])==spacing)
        yy,u=np.array(selected).T
        rotor=.5*(np.interp(radius,yy,u)+np.interp(-radius,yy,u))
        m1,m2,m3=map(float,moments(rotor,w));lam=4/m1
        row=dict(spacing_D=spacing,rotor_mean_velocity_m_s=.83*m1,TSR_local=lam)
        for field,m in [('Ct',m2),('Cp',m3)]:
            assert curves[field,0][0][0]<=lam<=curves[field,0][0][-1]
            predicted=float(np.interp(lam,*curves[field,0])*m);observed=float(np.interp(4,*curves[field,spacing]))
            row.update({field+'_predicted':predicted,field+'_measured':observed,field+'_error_percent':100*(predicted/observed-1)})
        out.append(row)
    (REPORT/'HIGH_TI_CONTROL.json').write_text(json.dumps(dict(rows=out,input='Part II isolated TI15 curve, measured Part I TI15 profiles; axisymmetric projection',
        scope='Offline performance-transfer control, not a coupled run or a calibrated viscosity model',
        extraction_allowance=.001,uncertainty='Plot extraction allowance is not experimental measurement uncertainty'),indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for ax,key in zip(axes,['Cp','Ct']):
        x=np.arange(2);ax.bar(x-.17,[r[key+'_measured'] for r in out],.34,label='Measured tandem',color='#333333')
        ax.bar(x+.17,[r[key+'_predicted'] for r in out],.34,label='Measured wake + isolated curve',color='#218e63')
        ax.set_xticks(x,['6D','10D']);ax.set(ylabel=key+' (nominal inflow)',title=key);ax.grid(axis='y',alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2)
    fig.suptitle('Independent high-turbulence control: TI15%, U=0.83 m/s')
    fig.savefig(REPORT/'HIGH_TI_CONTROL.png',dpi=160);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
