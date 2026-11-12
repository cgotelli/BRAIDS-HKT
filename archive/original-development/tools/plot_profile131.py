"""Plot non-overlapping worker phases; profiled timings are not clean speed claims."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation/_amr131'
paths=[OUT/'profile_before/cpu5m_0_baseline/PROFILE.json',
       OUT/'profile_after/cpu5m_0_candidate/PROFILE.json']
rows=[json.loads(p.read_text()) for p in paths]
phases=['load','construction','amr','output','checkpoint','step','other']
labels=['Load','Initial construction','AMR','Output','Checkpoint','Hydraulic stepping','Other']
colors=['#cbd5e1','#64748b','#ea580c','#a78bfa','#6d28d9','#2563eb','#94a3b8']
fig,ax=plt.subplots(figsize=(10,4.5))
left=[0.,0.]
for key,label,color in zip(phases,labels,colors):
    values=[r['phases'][key] for r in rows]
    ax.barh([1,0],values,left=left,label=label,color=color,height=.48)
    if key in ('amr','step'):
        for i,y in enumerate([1,0]):
            ax.text(left[i]+values[i]/2,y,f'{values[i]:.2f} s',ha='center',va='center',color='white',fontsize=10)
    left=[a+b for a,b in zip(left,values)]
ax.set_yticks([1,0],['Before','Alpha131'])
ax.set_xlim(0,17)
ax.set_xlabel('Instrumented worker time (s); phases do not overlap')
ax.set_title('5 m dynamic AMR: rebuilding still dominates',loc='left',weight='bold')
ax.spines[['top','right']].set_visible(False)
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.17),ncol=4,frameon=False)
fig.text(.01,.015,'Matched CFL 0.5: 297 hydraulic substeps in each run. Use AMR_SPEED.png for unprofiled speed measurements.',fontsize=9)
fig.tight_layout(rect=(0,.07,1,1))
fig.savefig(OUT/'AMR_PROFILE.png',dpi=180)
plt.close(fig)

