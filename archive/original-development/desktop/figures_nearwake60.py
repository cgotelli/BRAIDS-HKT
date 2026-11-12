from nearwake60 import *
import numpy as np
import matplotlib;matplotlib.use('Agg')
from matplotlib import pyplot as plt

def ledger_figure():
    f=np.load(REPORT/'WARM_OPERATOR_LEDGER.npz');xy=f['xy'];mask=(xy[:,1]==292.5)&(xy[:,0]>=475)&(xy[:,0]<=570);ids=np.flatnonzero(mask);ids=ids[np.argsort(xy[ids,0])]
    xx=(xy[ids,0]-500)/10;h=f['q'][0,ids]
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    axes[0,0].plot(xx,f['q'][1,ids]/h,'o-');axes[0,0].set(ylabel='Streamwise velocity (m/s)',title='Actual near-axis saved cell velocities')
    axes[0,1].plot(xx,1000*(h-20),'o-');axes[0,1].set(ylabel='Water surface above 20 m (mm)',title='Pressure response to the rotor')
    for key,label in [('actuator','Actuator'),('central_advection','Central advection'),('hkt_full_strain_faces','Viscous stress'),('total','Full spatial residual')]:
        axes[1,0].plot(xx,f[key][1,ids],label=label)
    for key,label in [('native_remainder','Native remainder'),('low_froude','Low-Froude correction'),('source_pressure','Source/pressure correction'),('pressure_interpolation','Pressure interpolation')]:
        axes[1,1].plot(xx,f[key][1,ids],label=label)
    for ax in axes[1]:ax.set(ylabel='Momentum rate (m²/s²)');ax.legend(fontsize=8)
    for ax in axes.ravel():ax.axvline(0,color='.5',ls=':');ax.set_xlabel('x/D');ax.grid(alpha=.2)
    fig.suptitle('Installed operator replay at 2,000 s; same state, no time advance. Friction is applied separately in time.')
    fig.savefig(REPORT/'WARM_OPERATOR_LEDGER.png',dpi=170);plt.close(fig)
if __name__=='__main__':ledger_figure()
