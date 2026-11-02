"""Resident hydraulic preparation for explicit recursive wet/dry trials.

Preparation mutates private RK operands, not the accepted state. Dry momentum
removal and the existing speed cap have separate area-integrated diagnostics.
No water or sediment is clipped and no synchronous/CPU fallback is performed.
"""
from pathlib import Path
import hashlib
import numpy as np


HERE = Path(__file__).resolve().parent
OPTIONS = ('--std=c++11','--fmad=false','--prec-div=true','--prec-sqrt=true','--ftz=false')


class ResidentWetPreparation:
    def __init__(self, area, levels, dry, speed_limit, *, source_path=None):
        area,levels = np.asarray(area),np.asarray(levels)
        if (area.ndim != 1 or not len(area) or levels.shape != area.shape or
                levels.dtype.kind not in 'iu' or np.any(levels < 0) or
                not np.isfinite(area).all() or np.any(area <= 0) or
                not np.isfinite([dry,speed_limit]).all() or dry <= 0 or speed_limit <= 0):
            raise ValueError('Finite positive hydraulic controls and cell geometry required')
        self.source_path = Path(source_path or HERE/'amr_bank_wet.cu').resolve()
        code = self.source_path.read_text(encoding='utf-8')
        if sum(line.strip()=='#define BRAIDS_BANK_WET_PREPARATION_ABI 1'
               for line in code.splitlines()) != 1:
            raise ValueError('Recursive wet preparation source ABI differs')
        self.source_sha256 = hashlib.sha256(self.source_path.read_bytes()).hexdigest()
        import cupy as cp
        self.cp=cp;self.nc=len(area);self.maximum_level=int(levels.max())
        self.area=cp.asarray(area,dtype=cp.float64)
        self.levels=cp.asarray(levels,dtype=cp.int32)
        self.dry=np.float64(dry);self.speed_limit=np.float64(speed_limit)
        self.impulse=cp.zeros((2,2),dtype=cp.float64)
        self.cell_impulse=cp.zeros((self.nc,2,2),dtype=cp.float64)
        self.cell_status=cp.zeros(self.nc,dtype=cp.int32)
        module=cp.RawModule(code=code,options=OPTIONS,name_expressions=('bw_prepare',))
        self.kernel=module.get_function('bw_prepare')

    def prepare(self,q,level,weight=1.,*,dry_only=False):
        """Prepare all traces; book only the advancing level's weighted impulse.

        Level -1 books all cells and is reserved for accepted-boundary dry-only
        cleanup. For SSPRK33, the private initial/a/b/final weights are
        1, 1/6, 2/3, 1 respectively. These are diagnostics, not external fluxes.
        """
        cp=self.cp
        if (not isinstance(q,cp.ndarray) or q.dtype != cp.float64 or
                q.shape != (3,self.nc) or not q.flags.c_contiguous or
                type(level) is not int or not -1 <= level <= self.maximum_level or
                type(dry_only) is not bool or (level == -1 and not dry_only) or
                not np.isfinite(weight) or weight < 0):
            raise ValueError('Invalid recursive hydraulic preparation input')
        self.kernel(((self.nc+127)//128,),(128,),
            (q,self.area,self.levels,np.int32(self.nc),np.int32(level),
             self.dry,self.speed_limit,np.int32(dry_only),self.cell_impulse,self.cell_status))
        if bool(cp.any(self.cell_status).item()):
            raise ValueError('Recursive hydraulic preparation rejected nonfinite or negative state')
        self.impulse += np.float64(weight)*cp.sum(self.cell_impulse,axis=0)
        if not bool(cp.all(cp.isfinite(self.impulse)).item()):
            raise ValueError('Recursive hydraulic preparation impulse is nonfinite')
        return self.cell_status
