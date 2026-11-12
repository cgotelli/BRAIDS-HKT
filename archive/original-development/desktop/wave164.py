"""Isolated bed-row flux candidates; no installed/public method selector."""
from pathlib import Path


def source(code,mode='matrix',audit=False):
    if mode not in ('matrix','hll','rusanov','bed_diagonal','morph_llf','capacity_diagonal'):
        raise ValueError(mode)
    density='law_params[13]' if 'law_params[13]/acceleration;' in code else '2650.'
    needle='  double next=(.5*(qs[0]+qs[1]-diss))*'+density+'/acceleration;'
    if code.count(needle)!=1:raise ValueError('Unexpected resolved wave layout')
    extra=''
    if audit:
        extra='''
  for(int k=0;k<4;++k){
   diagnostic[(3+k)*nf+i]=row[k];diagnostic[(7+k)*nf+i]=jump[k];
   diagnostic[(11+k)*nf+i]=-.5*row[k]*jump[k]*law_params[13]/acceleration;
   diagnostic[(15+k)*nf+i]=node[k];
  }
  diagnostic[19*nf+i]=.5*(qs[0]+qs[1])*law_params[13]/acceleration;
  diagnostic[20*nf+i]=s[1][3]-s[0][3];
  diagnostic[21*nf+i]=s[0][0];diagnostic[22*nf+i]=s[1][0];
'''
    if mode=='hll':
        extra+='''
  // Bed row of the two-wave flux: hydraulic f-wave solver is unchanged.
  const double sl=fmin(0.,node[0]),sr=fmax(0.,node[3]);
  next=(sr*qs[0]-sl*qs[1]+sl*sr*(s[1][3]-s[0][3]))/(sr-sl)*law_params[13]/acceleration;
'''
    elif mode=='rusanov':
        extra+='''
  next=.5*(qs[0]+qs[1]-speed*(s[1][3]-s[0][3]))*law_params[13]/acceleration;
'''
    elif mode=='bed_diagonal':
        extra+='''
  // Bed diagonal of |A| = sign(A) A; do not inject hydraulic-state
  // dissipation into the independently updated Exner row.
  const double bed_speed=fmax(0.,row[1]*gh);
  next=.5*(qs[0]+qs[1]-bed_speed*(s[1][3]-s[0][3]))*law_params[13]/acceleration;
'''
    elif mode=='morph_llf':
        extra+='''
  const double bed_speed=fmin(fabs(roots[0]),fmin(fabs(roots[1]),fabs(roots[2])));
  next=.5*(qs[0]+qs[1]-bed_speed*(s[1][3]-s[0][3]))*law_params[13]/acceleration;
'''
    elif mode=='capacity_diagonal':
        extra+='''
  const double bed_speed=fmax(0.,row[1]*gh);
  next=face[i]*nx+face[nf+i]*ny-.5*bed_speed*(s[1][3]-s[0][3])*law_params[13]/acceleration;
'''
    extra=extra.replace('law_params[13]',density)
    return '// ISOLATED_WAVE164 '+mode+'\n'+code.replace(needle,needle+extra)


def install(mode,audit=False):
    from braids_core.sediment import river_capacity
    from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine
    previous=river_capacity.wave_source
    river_capacity.wave_source=lambda **kw:source(previous(**kw),mode,audit)
    # Keep the authenticated loader for all retained kernels; compile only
    # this explicitly identified experiment with the normal cache/compiler.
    portable=WaveEngine.__init__
    compile_init=getattr(portable,'__wrapped__',portable)
    def init(self,*args,**kw):
        p=kw.get('source')
        candidate=p is not None and Path(p).read_text().startswith('// ISOLATED_WAVE164 ')
        return (compile_init if candidate else portable)(self,*args,**kw)
    WaveEngine.__init__=init
