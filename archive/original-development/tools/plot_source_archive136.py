"""Render measured archive-only preparation costs; no hydraulic speed claims."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser();p.add_argument('--sweep',type=Path,required=True)
    p.add_argument('--abba',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sweep=json.loads(a.sweep.read_text());abba=json.loads(a.abba.read_text())
    groups=[[r['seconds'] for r in abba['results'] if r['workers']==w] for w in (0,16)]
    medians=[statistics.median(values) for values in groups]
    fig,ax=plt.subplots(1,2,figsize=(10.8,4.6))
    colors=['#606d80','#118577']
    ax[0].bar([0,1],medians,color=colors,width=.58)
    for x,values in enumerate(groups):
        ax[0].plot([x+(j-(len(values)-1)/2)*.055 for j in range(len(values))],values,
                   'o',color='#20252e',markersize=4)
        ax[0].text(x,medians[x]+.14,f'{medians[x]:.3f} s',ha='center',fontweight='bold')
    ax[0].set_xticks([0,1],['Released serial','16 I/O threads'])
    ax[0].set_ylim(0,max(max(v) for v in groups)*1.18)
    ax[0].set_ylabel('Source-archive preparation (s)')
    ax[0].set_title(f'Unmodified-reference ABBA: {medians[0]/medians[1]:.2f}× faster')
    workers=[1,4,8,16]
    for phase,label,color in [('copy','Copy + initial hash','#497bb3'),('verify','Full source/copy verification','#b05d39')]:
        values=[sweep['summary'][str(w)][phase+'_median_s'] for w in workers]
        # The very first serial copy was a cold-start outlier. Keep it in raw
        # evidence, but show its warm counterpart rather than an inflated median.
        if phase=='copy':
            values[0]=[s['seconds'] for r in sweep['results'][1:] if r['workers']==1
                       for s in r['phases'] if s['phase']=='copy'][0]
        ax[1].plot(workers,values,'o-',label=label,color=color)
    ax[1].set_xticks(workers);ax[1].set_ylim(bottom=0)
    ax[1].set_xlabel('Independent I/O workers');ax[1].set_ylabel('Per-phase time (s)')
    ax[1].set_title('All byte reads and integrity checks retained')
    ax[1].legend(frameon=False,fontsize=9)
    for axis in ax:
        axis.spines[['top','right']].set_visible(False)
        axis.grid(axis='y',alpha=.2);axis.set_axisbelow(True)
    fig.suptitle('Isolated prototype: 788 sources · 796 byte-identical archived files',fontweight='bold')
    fig.text(.5,.015,'Archive-only measurements on this PC; not a whole-simulation speedup. Serial copy excludes initial cold outlier.',
             ha='center',fontsize=8,color='#555555')
    fig.tight_layout(rect=(0,.045,1,.94));a.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output,dpi=170);plt.close(fig)


if __name__=='__main__':main()
