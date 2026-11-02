"""Same-call reuse of the grain-independent mean-state wave coefficients.

The physical wave source stays canonical. Generate one shared per-face body,
with ordinary and cached entry points; both execute the same arithmetic. No
state survives a class batch as an admissible cache hit.
"""
import re

FIELDS=('h','u','v','gh','pressure_jump','sp2','zrough','cmean','velocity_factor','cwork')

def source(code):
    match=re.search(r'ENTRY void matrix_bed_faces\((.*?)\)\{\n LOOP\(nf\)\{(.*)\n \}\n\}\s*$',code,re.S)
    if match is None:raise ValueError('Unrecognized mean-state wave source')
    formals,body=match.groups()
    start=body.index('  double h=.5*')
    end=body.index('  double mag=',start)
    common=body[start:end]
    # MPM replaces mag with its mixture ripple factor before the magnitude.
    if '  double dm90=' in common:
        cut=common.index('  double dm90=');end=start+cut;common=common[:cut]
    old='double h=.5*(s[0][0]+s[1][0]),rt0=sqrt(s[0][0]),rt1=sqrt(s[1][0]);'
    assert common.count(old)==1
    common=common.replace(old,'h=.5*(s[0][0]+s[1][0]); double rt0=sqrt(s[0][0]),rt1=sqrt(s[1][0]);')
    for name in FIELDS[1:]:common=common.replace('double '+name+'=',name+'=')
    # ve is only used in the EH capacity; leave it outside the common block.
    old=',ve=sqrt(sp2)*velocity_factor;'
    assert common.count(old)==1
    common=common.replace(old,';')
    read=''.join(f'{v}=cache[{k}*nf+i];' for k,v in enumerate(FIELDS))
    write=''.join(f'cache[{k}*nf+i]={v};' for k,v in enumerate(FIELDS))
    replacement=('  double '+','.join(FIELDS)+';\n'
        +'  if(reuse){'+read+'}\n  else {\n'+common
        +'   if(cache){'+write+'}\n  }\n  double ve=sqrt(sp2)*velocity_factor;\n')
    body=body[:start]+replacement+body[end:]
    # All continue statements in this source target the original face loop.
    # Internal polynomial loops contain break, not continue.
    body=body.replace('continue;','return;')
    names=[re.search(r'([A-Za-z_]\w*)\s*$',arg).group(1) for arg in formals.split(',')]
    args=','.join(names)
    helper='LOCAL void class_mean_face(int i,'+formals+',int reuse,double*cache){'+body+'\n}\n'
    plain='ENTRY void matrix_bed_faces('+formals+'){LOOP(nf){class_mean_face(i,'+args+',0,0);}}\n'
    cached='ENTRY void matrix_bed_faces_cached('+formals+',int reuse,double*cache){LOOP(nf){class_mean_face(i,'+args+',reuse,cache);}}\n'
    return code[:match.start()]+helper+plain+cached

def enable(engine,xp,nf):
    """Private scratch, not accepted state and not a public snapshot array."""
    engine._class_shared=xp.empty((len(FIELDS),nf),dtype=xp.float64)

def bind(engine):
    engine.fn['matrix_bed_faces']=engine.fn['matrix_bed_faces_cached']

NAMES=('wave_state_gradients','matrix_bed_faces_cached')
