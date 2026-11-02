"""Cell averages of the same continuous DEM surface used by inlet sections.

Source elevations are nodes at pixel centres. Interpolate bilinearly between
them, with nearest-value extension through the outer half pixel, matching
``_RasterSampler``. This is terrain integration, not computational-cell splitting.
Raster-valued material properties retain their separate pixel-average path.
"""
import math
import numpy as np
from osgeo import ogr,osr
from braids_core.boundaries.gis import _RasterSampler
from braids_core.boundaries.cut_cells import _parts


def linear_weights(lo,hi,size):
    """Exact integrals of linear nodal basis functions in pixel coordinates."""
    if not 0<=lo<hi<=size:raise ValueError('Integration interval lies outside the raster')
    left=max(lo,.5);right=min(hi,size-.5)
    first=max(0,math.floor(left-.5));last=min(size-1,math.ceil(right-.5))
    ids=np.arange(first,max(first,last)+1)
    def integral(t):
        t=np.clip(t,-1.,1.)
        return np.where(t<0,.5*(t+1.)**2,1.-.5*(1.-t)**2)
    weights=np.zeros(len(ids)) if right<=left else integral(right-ids-.5)-integral(left-ids-.5)
    if lo<.5:
        if ids[0]!=0:ids=np.r_[0,ids];weights=np.r_[0.,weights]
        weights[0]+=min(hi,.5)-lo
    if hi>size-.5:
        if ids[-1]!=size-1:ids=np.r_[ids,size-1];weights=np.r_[weights,0.]
        weights[-1]+=hi-max(lo,size-.5)
    keep=weights>0
    return ids[keep],weights[keep]


def polygon_moments(geometry):
    """Integrals of 1, x, y and xy on a local polygon, including its holes."""
    total=np.zeros(4)
    for polygon in _parts(geometry,ogr.wkbPolygon):
        for k in range(polygon.GetGeometryCount()):
            xy=np.asarray(polygon.GetGeometryRef(k).GetPoints())[:,:2]
            x,y=xy.T;xn=np.roll(x,-1);yn=np.roll(y,-1);cross=x*yn-xn*y
            area=.5*cross.sum()
            if area==0:continue
            moment=np.array([area,((x+xn)*cross).sum()/6.,((y+yn)*cross).sum()/6.,
                ((2*x*y+x*yn+xn*y+2*xn*yn)*cross).sum()/24.])
            total+=(1 if k==0 else -1)*math.copysign(1.,area)*moment
    return total


def polygon(vertices,holes=()):
    result=ogr.Geometry(ogr.wkbPolygon)
    for points in (vertices,*holes):
        ring=ogr.Geometry(ogr.wkbLinearRing)
        for x,y in points:ring.AddPoint_2D(float(x),float(y))
        ring.CloseRings();result.AddGeometry(ring)
    return result


class RasterSurface:
    def __init__(self,path):
        sample=_RasterSampler(path,allow_undeclared_metres=True)
        self.sample=sample;self.transform=sample.transform
        _,sx,rx,_,ry,sy=self.transform
        if sx<=0 or sy>=0 or rx!=0 or ry!=0:raise ValueError('Terrain integration requires a north-up raster')
        raw=np.asarray(sample.band.ReadAsArray(),dtype=float)
        valid=np.isfinite(raw)
        if sample.nodata is not None:valid &= raw!=sample.nodata
        if sample.mask is not None:valid &= sample.mask.ReadAsArray()!=0
        self.values=raw*sample.scale+sample.offset
        self.valid=valid&np.isfinite(self.values)
        self.height,self.width=raw.shape;self.pixel_area=-sx*sy
        self.xweights={};self.yweights={};self._quadrature={}

    def weights(self,lo,hi,axis):
        cache=self.xweights if axis==0 else self.yweights
        key=(float(lo),float(hi))
        if key not in cache:cache[key]=linear_weights(*key,self.width if axis==0 else self.height)
        return cache[key]

    def rectangle_average(self,xlo,xhi,ylo,yhi):
        cols,wx=self.weights(xlo,xhi,0);rows,wy=self.weights(ylo,yhi,1)
        index=np.ix_(rows,cols)
        if not self.valid[index].all():raise ValueError('DEM interpolation touches NoData/invalid source pixels')
        return float(np.sum(self.values[index]*wy[:,None]*wx[None,:])/((xhi-xlo)*(yhi-ylo)))

    def cartesian_averages(self,grid):
        """Integrate complete grid rectangles with separable DEM weights.

        The native background grid is deliberately snapped beyond a polygon's
        bounds.  A corner rectangle can therefore extend beyond the DEM even
        though its *retained cut-cell polygon* is fully covered.  Leave those
        rectangles marked invalid so ``surface_averages`` uses exact polygon
        integration for them, instead of rejecting the whole preparation grid.
        """
        gt=self.transform
        x_edges=(grid.xlower_m+np.arange(grid.nx+1)*grid.dx_m-gt[0])/gt[1]
        y_edges=(grid.ylower_m+np.arange(grid.ny+1)*grid.dy_m-gt[3])/gt[5]
        x_covered=((np.minimum(x_edges[:-1],x_edges[1:])>=-1e-8)&
                   (np.maximum(x_edges[:-1],x_edges[1:])<=self.width+1e-8))
        y_covered=((np.minimum(y_edges[:-1],y_edges[1:])>=-1e-8)&
                   (np.maximum(y_edges[:-1],y_edges[1:])<=self.height+1e-8))
        x_edges=np.clip(x_edges,0.,self.width)
        y_edges=np.clip(y_edges,0.,self.height)
        covered_columns=np.flatnonzero(x_covered)
        along_x=np.empty((self.height,grid.nx));valid_x=np.zeros((self.height,grid.nx),dtype=bool)
        for i in range(grid.nx):
            if not x_covered[i]:continue
            lo,hi=sorted(x_edges[i:i+2]);cols,weights=self.weights(lo,hi,0)
            along_x[:,i]=self.values[:,cols]@weights/(hi-lo)
            valid_x[:,i]=self.valid[:,cols].all(axis=1)
        result=np.full((grid.ny,grid.nx),np.nan);valid=np.zeros((grid.ny,grid.nx),dtype=bool)
        for j in range(grid.ny):
            if not y_covered[j]:continue
            lo,hi=sorted(y_edges[j:j+2]);rows,weights=self.weights(lo,hi,1)
            index=np.ix_(rows,covered_columns)
            result[j,covered_columns]=weights@along_x[index]/(hi-lo)
            valid[j,covered_columns]=valid_x[index].all(axis=0)
        return result,valid

    def cell_bounds(self,cell):
        gt=self.transform;xy=np.asarray(cell.vertices_xy,dtype=float)
        uv=np.column_stack(((xy[:,0]-gt[0])/gt[1],(xy[:,1]-gt[3])/gt[5]))
        lo=uv.min(axis=0);hi=uv.max(axis=0);size=np.array([self.width,self.height])
        if np.any(lo<-1e-8) or np.any(hi>size+1e-8):raise ValueError(f'DEM does not cover cell {cell.cell_id}')
        uv=np.clip(uv,0,size);lo=uv.min(axis=0);hi=uv.max(axis=0)
        box_area=float(np.prod(hi-lo)*self.pixel_area)
        rectangular=(not cell.holes_xy and math.isclose(cell.area_m2,box_area,rel_tol=2e-12,abs_tol=2e-12*self.pixel_area)
            and all((abs(x-lo[0])<1e-11 or abs(x-hi[0])<1e-11) and
                    (abs(y-lo[1])<1e-11 or abs(y-hi[1])<1e-11) for x,y in uv))
        return lo,hi,box_area,rectangular

    def irregular_averages(self,cells,*,threads=1,cancelled=None,progress=None):
        from braids_core.boundaries.raster_quadrature import RasterQuadrature
        if cancelled and cancelled():raise InterruptedError('BRAIDS preparation cancelled')
        if threads not in self._quadrature:self._quadrature[threads]=RasterQuadrature(threads)
        return self._quadrature[threads].integrate(self,cells,cancelled=cancelled,progress=progress)

    def cell_average(self,cell):
        lo,hi,area,rectangular=self.cell_bounds(cell)
        if rectangular:
            return self.rectangle_average(lo[0],hi[0],lo[1],hi[1]),area,True
        values,areas=self.irregular_averages((cell,))
        return float(values[0]),float(areas[0]),False


def surface_averages(mesh,dem_path,progress=None,cancelled=None,*,threads=1):
    if type(threads) is not int or not 1<=threads<=256:raise ValueError('Terrain preparation threads must be an integer in 1..256')
    if cancelled and cancelled():raise InterruptedError('BRAIDS preparation cancelled')
    surface=RasterSurface(dem_path);crs=osr.SpatialReference();crs.ImportFromWkt(mesh.crs_wkt)
    if not surface.sample.spatial.IsSame(crs):raise ValueError('DEM and polygon CRS must match')
    values=np.empty(len(mesh.cells));errors=[];rectangles=0;cropped=[]
    grid=getattr(mesh,'grid',None)
    if grid is None:
        rectangle_values=rectangle_valid=None;nominal=None
    else:
        rectangle_values,rectangle_valid=surface.cartesian_averages(grid)
        nominal=grid.dx_m*grid.dy_m
    for k,cell in enumerate(mesh.cells):
        if cancelled and cancelled():raise InterruptedError('BRAIDS preparation cancelled')
        xlo=(0. if grid is None else grid.xlower_m+cell.i*grid.dx_m)
        ylo=(0. if grid is None else grid.ylower_m+cell.j*grid.dy_m)
        full=(grid is not None and not cell.holes_xy and
              math.isclose(cell.area_m2,nominal,rel_tol=2e-12,abs_tol=2e-12*nominal)
              and all((math.isclose(x,xlo,abs_tol=1e-10) or math.isclose(x,xlo+grid.dx_m,abs_tol=1e-10)) and
                      (math.isclose(y,ylo,abs_tol=1e-10) or math.isclose(y,ylo+grid.dy_m,abs_tol=1e-10))
                      for x,y in cell.vertices_xy))
        if full:
            if not rectangle_valid[cell.j,cell.i]:
                raise ValueError(f'DEM interpolation touches NoData/invalid source pixels under cell {cell.cell_id}')
            values[k]=rectangle_values[cell.j,cell.i];area=nominal;rectangular=True
        else:
            lo,hi,area,rectangular=surface.cell_bounds(cell)
            if rectangular:values[k]=surface.rectangle_average(lo[0],hi[0],lo[1],hi[1])
            else:cropped.append(k);continue
        error=area-cell.area_m2
        if abs(error)>max(1e-7,1e-9*cell.area_m2):raise ValueError(f'Terrain quadrature does not cover cell {cell.cell_id}')
        errors.append(error);rectangles+=int(rectangular)
        if progress and (k%max(1,len(mesh.cells)//100)==0 or k+1==len(mesh.cells)):progress(40*(k+1)/len(mesh.cells))
    if cropped:
        cells=tuple(mesh.cells[k] for k in cropped)
        averages,areas=surface.irregular_averages(cells,threads=threads,cancelled=cancelled,
            progress=None if progress is None else lambda p:progress(40+.6*p))
        for k,average,area in zip(cropped,averages,areas):
            error=area-mesh.cells[k].area_m2
            if abs(error)>max(1e-7,1e-9*mesh.cells[k].area_m2):raise ValueError(f'Terrain quadrature does not cover cell {mesh.cells[k].cell_id}')
            values[k]=average;errors.append(error)
    if cancelled and cancelled():raise InterruptedError('BRAIDS preparation cancelled')
    if progress:progress(100.)
    return values,{'method':'exact cell integral of bilinear elevations at source pixel centres',
        'outer_half_pixel':'nearest-node extension, matching inlet sections',
        'horizontal_units':'metres from the projected DEM/domain CRS',
        'vertical_units':'metres',
        'declared_vertical_units':surface.sample.declared_vertical_units,
        'vertical_units_source':surface.sample.vertical_units_source,
        'vertical_datum':'source DEM elevations retained; no independent datum shift',
        'analytic_rectangle_cells':rectangles,'intersection_cells':len(mesh.cells)-rectangles,
        'maximum_partition_error_m2':max(map(abs,errors)),
        'minimum_bed_m':float(values.min()),'maximum_bed_m':float(values.max()),
        'gdal_scale':surface.sample.scale,'gdal_offset':surface.sample.offset}
