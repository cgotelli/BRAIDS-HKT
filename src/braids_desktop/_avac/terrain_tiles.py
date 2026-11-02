"""Visible-only DEM color tiles. Never changes the source raster or solver input."""
from collections import OrderedDict
import math
import numpy as np
from PySide6.QtCore import QRectF,Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QGraphicsItem


def level_samples(raster,sx,sy,x0,x1,y0,y1):
    """Sample display-block centres; stride one uses exact native DEM samples.

    Indices refer to the north-up display lattice. Independent X/Y steps also
    support non-square DEM pixels. A partial final block stays at the real DEM
    edge, not stretched over the whole image. This is not scientific resampling.
    """
    rows,cols=raster.z.shape
    cx=(np.arange(x0,x1)*sx+np.minimum(np.arange(x0,x1)*sx+sx,cols))*.5
    cy=(np.arange(y0,y1)*sy+np.minimum(np.arange(y0,y1)*sy+sy,rows))*.5
    u=cx-.5;v=cy-.5
    lo_x=np.floor(u).astype(int);hi_x=np.ceil(u).astype(int)
    lo_y=np.floor(v).astype(int);hi_y=np.ceil(v).astype(int)
    wx=(u-lo_x)[None,:];wy=(v-lo_y)[:,None]
    north=raster.z[::-1]
    a=north[np.ix_(lo_y,lo_x)]*(1-wx)+north[np.ix_(lo_y,hi_x)]*wx
    b=north[np.ix_(hi_y,lo_x)]*(1-wx)+north[np.ix_(hi_y,hi_x)]*wx
    return a*(1-wy)+b*wy,cx,cy


class TerrainTiles(QGraphicsItem):
    """Render/cache only visible tiles at the actual device/export pixel scale."""
    tile_size=256

    def __init__(self,raster,limits,*,palette='Terrain',max_bytes=64*1024*1024):
        super().__init__()
        self.raster=raster;meta=raster.metadata
        self.dx=float(meta['cellsize']);self.dy=float(meta.get('cellsize_y',self.dx))
        self.rows,self.cols=raster.z.shape
        self._bounds=QRectF(0,0,self.cols*self.dx,self.rows*self.dy)
        self.cache=OrderedDict();self.cache_bytes=0;self.max_bytes=max_bytes
        self.tiles_built=0;self.last_stride=(1,1);self.last_tile_keys=[]
        self._style=None;self.configure(limits,palette=palette)
        self.setFlag(QGraphicsItem.ItemUsesExtendedStyleOption,True)
        self.setToolTip(f'Terrain: native DEM pixels {self.dx:g} × {self.dy:g} m. '
                        'Zoom in for native detail; overview tiles are display-only.')

    def boundingRect(self):return self._bounds

    def configure(self,limits,*,palette='Terrain',zero_transparent=False):
        style=(tuple(map(float,limits)),palette,bool(zero_transparent))
        if style!=self._style:
            self._style=style;self.cache.clear();self.cache_bytes=0;self.update()

    def stride_for(self,transform):
        # deviceTransform includes high-DPI scaling and image-export scale.
        px=math.hypot(transform.m11(),transform.m12())*self.dx
        py=math.hypot(transform.m21(),transform.m22())*self.dy
        def step(pixels,count):
            level=max(0,math.floor(math.log2(1/max(pixels,1e-30))))
            return 2**min(level,max(0,math.ceil(math.log2(count))))
        return step(px,self.cols),step(py,self.rows)

    def tile(self,sx,sy,tx,ty):
        key=(sx,sy,tx,ty)
        if key in self.cache:
            self.cache.move_to_end(key);return self.cache[key]
        from .map_view import colour_image
        nx=math.ceil(self.cols/sx);ny=math.ceil(self.rows/sy);size=self.tile_size
        x0=tx*size;y0=ty*size;x1=min(x0+size,nx);y1=min(y0+size,ny)
        # One display-sample halo makes the hillshade identical on both sides
        # of every tile boundary. Coordinates handle partial edge blocks too.
        hx0=max(0,x0-1);hy0=max(0,y0-1);hx1=min(nx,x1+1);hy1=min(ny,y1+1)
        values,cx,cy=level_samples(self.raster,sx,sy,hx0,hx1,hy0,hy1)
        limits,palette,zero=self._style
        image,_,_=colour_image(values[::-1],terrain=True,
            cell=(-cy[::-1]*self.dy,cx*self.dx),limits=limits,palette=palette,zero_transparent=zero)
        image=image.copy(x0-hx0,y0-hy0,x1-x0,y1-y0)
        self.tiles_built+=1;self.cache[key]=image;self.cache_bytes+=image.sizeInBytes()
        while self.cache_bytes>self.max_bytes and len(self.cache)>1:
            _,old=self.cache.popitem(last=False);self.cache_bytes-=old.sizeInBytes()
        return image

    def paint(self,painter,option,widget=None):
        visible=option.exposedRect.intersected(self._bounds)
        # QGraphicsScene.render (exports) can report the whole item as exposed
        # even when its painter is clipped to a small map window.
        if painter.hasClipping():visible=visible.intersected(painter.clipBoundingRect())
        if visible.isEmpty():return
        sx,sy=self.stride_for(painter.deviceTransform());self.last_stride=(sx,sy)
        width=self.tile_size*sx*self.dx;height=self.tile_size*sy*self.dy
        x0=max(0,math.floor(visible.left()/width));x1=min(math.ceil(self.cols/(self.tile_size*sx)),math.ceil(visible.right()/width))
        y0=max(0,math.floor(visible.top()/height));y1=min(math.ceil(self.rows/(self.tile_size*sy)),math.ceil(visible.bottom()/height))
        painter.save();painter.setClipRect(self._bounds,Qt.IntersectClip)
        painter.setRenderHint(QPainter.SmoothPixmapTransform,False)
        painter.setRenderHint(QPainter.Antialiasing,False)
        self.last_tile_keys=[]
        for ty in range(y0,y1):
            for tx in range(x0,x1):
                image=self.tile(sx,sy,tx,ty);self.last_tile_keys.append((sx,sy,tx,ty))
                # Constant physical pixel spacing; only the partial outermost
                # block is clipped. Never stretch the raster to fit ceil(N/s).
                target=QRectF(tx*width,ty*height,image.width()*sx*self.dx,image.height()*sy*self.dy)
                painter.drawImage(target,image,QRectF(image.rect()))
        painter.restore()
