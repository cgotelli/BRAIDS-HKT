"""Compiled, cancellable exact terrain integration; no solver/backend selection.

Signed clipped rings retain holes and concave/disconnected intersections.
Local unit-pixel moments avoid cancellation at large projected coordinates.
The existing portable WaveEngine loader records this preparation-only DLL so
QGIS users do not need a compiler.
"""
from pathlib import Path
import math
import numpy as np
from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine


class RasterQuadrature:
    def __init__(self,threads=1):
        if type(threads) is not int or not 1<=threads<=256:
            raise ValueError('Terrain preparation threads must be an integer in 1..256')
        self.engine=WaveEngine('cpu',threads,source=Path(__file__).with_suffix('.h'),
                               names=('raster_integrals',))

    def integrate(self,surface,cells,*,cancelled=None,progress=None):
        def check():
            if cancelled and cancelled():raise InterruptedError('BRAIDS preparation cancelled')
        check();cell_rings=[0];ring_points=[0];points=[];jobs=[]
        gt=surface.transform;size=np.array([surface.width,surface.height])
        for i,cell in enumerate(cells):
            check()
            for ring in (cell.vertices_xy,*cell.holes_xy):
                xy=np.asarray(ring,dtype=float)
                if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<3 or not np.isfinite(xy).all():
                    raise ValueError(f'Invalid terrain polygon in cell {cell.cell_id}')
                uv=np.column_stack(((xy[:,0]-gt[0])/gt[1],(xy[:,1]-gt[3])/gt[5]))
                if np.any(uv<-1e-8) or np.any(uv>size+1e-8):
                    raise ValueError(f'DEM does not cover cell {cell.cell_id}')
                uv=np.clip(uv,0,size)
                if len(ring_points)==cell_rings[-1]+1:outer=uv
                points.extend(uv);ring_points.append(len(points))
            cell_rings.append(len(ring_points)-1)
            lo=outer.min(axis=0);hi=outer.max(axis=0)
            c0=max(-1,math.floor(lo[0]-.5));c1=min(surface.width-1,math.ceil(hi[0]-.5)-1)+1
            r0=max(-1,math.floor(lo[1]-.5));r1=min(surface.height-1,math.ceil(hi[1]-.5)-1)+1
            # Bound each job to <=4096 quadrature pieces; allow cancellation
            # between batches even when a single retained cell spans a large DEM.
            for r in range(r0,r1,64):
                for c in range(c0,c1,64):jobs.append((i,c,min(c+64,c1),r,min(r+64,r1)))
        if len(points)>np.iinfo(np.int32).max:raise ValueError('Terrain polygon inventory is too large')
        cr=np.asarray(cell_rings,dtype=np.int32);rp=np.asarray(ring_points,dtype=np.int32)
        xy=np.ascontiguousarray(points,dtype=float).reshape(-1,2)
        jobs=np.asarray(jobs,dtype=np.int32).reshape(-1,5)
        z=np.ascontiguousarray(surface.values,dtype=float)
        valid=np.ascontiguousarray(surface.valid,dtype=np.uint8)
        totals=np.zeros((len(cells),2))
        for start in range(0,len(jobs),64):
            check();batch=np.ascontiguousarray(jobs[start:start+64]);n=len(batch)
            result=np.empty((n,2));error=np.zeros(n,dtype=np.int32)
            self.engine.launch('raster_integrals',n,n,surface.width,surface.height,
                float(surface.pixel_area),cr,rp,xy,batch,z,valid,result,error)
            if error.any():
                cell=cells[int(batch[np.flatnonzero(error)[0],0])]
                raise ValueError(f'DEM interpolation touches NoData/invalid source pixels under cell {cell.cell_id}')
            # Canonical job order, independent of OpenMP scheduling/thread count.
            for spec,part in zip(batch,result):totals[spec[0]]+=part
            if progress:progress(100*min(start+64,len(jobs))/max(1,len(jobs)))
        check()
        bad=np.flatnonzero((totals[:,1]<=0)|~np.isfinite(totals).all(axis=1))
        if len(bad):raise ValueError(f'DEM overlap is empty/invalid for cell {cells[int(bad[0])].cell_id}')
        return totals[:,0]/totals[:,1],totals[:,1]
