"""Least-change bounded eta/bed gradients, paired native/CUDA implementation.

Project the LS gradient onto the intersection of the face extrapolation bounds.
Unlike one scalar multiplier, an active transverse bound need not suppress an
independent longitudinal derivative. Closed walls provide no exterior terrain
constraint; their velocity reflection and pressure flux are unchanged.

At most 32 faces use the bounded local projection workspace. More complex
agglomerates retain the conservative scalar limiter, explicitly on both
backends. This is a geometry capacity guard, not a selectable numerical mode.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
METHOD = 'directional_eta_bed_neighbor_bounds_wall_reflection_v1'


def native_source(code):
    marker = '   do m=1,nprimitive\n    grad(m,:,c)=grad(m,:,c)*max(0.d0,theta(m))'
    if code.count(marker) != 3:
        raise ValueError('Expected all three native hydraulic reconstruction routes')
    code = code.replace(marker, '   call braids_project_cell(c,v,grad,vlo,vhi,theta)\n'+marker)
    end = code.rindex('\nend module')
    return code[:end]+'\n'+(HERE/'directional_hydraulic_limiter.f90.inc').read_text()+code[end:]


def cuda_source(code):
    marker = '            for(int m=0;m<4;++m) {gx[m]*=fmax(0.,theta[m]);gy[m]*=fmax(0.,theta[m]);}'
    if code.count(marker) != 1:
        raise ValueError('Expected one CUDA hydraulic reconstruction loop')
    helper, insert = (HERE/'directional_hydraulic_limiter.cu.inc').read_text().split('// CELL_INSERT\n')
    return helper+code.replace(marker, insert+marker+'''
            if(project_candidate){gx[0]=px[0];gy[0]=py[0];gx[3]=px[1];gy[3]=py[1];}
''')
