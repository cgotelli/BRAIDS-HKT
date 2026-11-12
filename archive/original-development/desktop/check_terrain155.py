"""Read-only real-DEM display comparison and clean cold/cached paint timings."""
import hashlib,json,math,os,sys,time
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_terrain155';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT))
import numpy as np
from PySide6.QtCore import QRectF,Qt
from PySide6.QtGui import QImage,QPainter,QPixmap,QTransform
from PySide6.QtWidgets import QApplication,QGraphicsScene,QGraphicsPixmapItem
from braids_desktop._avac.geo import read_terrain
from braids_desktop._avac.map_view import colour_image
from braids_desktop._avac.terrain_tiles import TerrainTiles
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(scene,source,width=1000,height=750):
    image=QImage(width,height,QImage.Format_RGBA8888);image.fill(0)
    painter=QPainter(image);scene.render(painter,QRectF(0,0,width,height),source,Qt.IgnoreAspectRatio);painter.end()
    return image


def rgba(image):
    return np.asarray(image.constBits()).reshape(image.height(),image.bytesPerLine())[:,:image.width()*4].reshape(image.height(),image.width(),4).copy()


app=QApplication.instance() or QApplication([])
project=Path('C:/Users/STREEM/Desktop/01obstacle/River.braids.json')
doc=json.loads(project.read_text(encoding='utf-8'))
dem=project.parent/doc['data_folder']/doc['project']['terrain']['dem_path']
terrain=read_terrain(dem);raster=terrain.raster;meta=raster.metadata
before=hashlib.sha256(memoryview(raster.z)).hexdigest()
# Same real river/levee window for both views, no simulation or geometry edits.
east,north=2571335.,1107390.;width,height=120.,90.
source=QRectF(east-width/2-meta['xmin'],meta['ymax']-(north+height/2),width,height)
row=(raster.y>=north-height/2)&(raster.y<=north+height/2)
col=(raster.x>=east-width/2)&(raster.x<=east+width/2)
limits=tuple(map(float,np.nanpercentile(raster.z[np.ix_(row,col)],[1,99])))
stride=max(1,math.ceil(max(raster.z.shape)/1600))
old_image,_,_=colour_image(raster.z[::stride,::stride],terrain=True,cell=meta['cellsize']*stride,limits=limits,palette='Grayscale')
old=QGraphicsPixmapItem(QPixmap.fromImage(old_image));old.setTransform(QTransform.fromScale((meta['xmax']-meta['xmin'])/old_image.width(),(meta['ymax']-meta['ymin'])/old_image.height()))
old_scene=QGraphicsScene();old_scene.addItem(old);old_render=render(old_scene,source)
item=TerrainTiles(raster,limits,palette='Grayscale');scene=QGraphicsScene();scene.addItem(item)
times={};stages={}
for name,rect in [('overview',item.boundingRect()),('zoom',source),('pan',source.translated(8,4))]:
    count=item.tiles_built;start=time.perf_counter();image=render(scene,rect)
    cold=time.perf_counter()-start;warm=[]
    for _ in range(5):
        start=time.perf_counter();render(scene,rect);warm.append(time.perf_counter()-start)
    times[name]=dict(first_paint_s=cold,cached_median_s=float(np.median(warm)))
    stages[name]=dict(stride=item.last_stride,visible_tiles=len(item.last_tile_keys),new_tiles=item.tiles_built-count,cache_bytes=item.cache_bytes)
new_render=render(scene,source);assert item.last_stride==(1,1)
fig,axes=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
for ax,image,title in zip(axes,(old_render,new_render),('Previous fixed overview: 2.5 m display','Zoom-dependent tiles: native 0.5 m detail')):
    ax.imshow(rgba(image),extent=(-width/2,width/2,-height/2,height/2),interpolation='nearest')
    ax.set_title(title);ax.set_xlabel(f'Local x from {east:.0f} m');ax.set_ylabel(f'Local y from {north:.0f} m')
fig.suptitle('Same DEM, extent, palette and color limits — visualization only')
fig.savefig(OUT/'TERRAIN_DETAIL.png',dpi=160);plt.close(fig)
assert before==hashlib.sha256(memoryview(raster.z)).hexdigest()
evidence=dict(dem=str(dem),dem_sha256=terrain.source_hash,source_pixels=[item.cols,item.rows],native_pixel_m=[item.dx,item.dy],
    former_display_stride=stride,new_zoom_stride=item.last_stride,display_limits_m=limits,render_timings=times,stages=stages,
    source_array_unchanged=True,solver_runtime_sha256=hashlib.sha256((ROOT/'desktop/runtime/RUNTIME.json').read_bytes()).hexdigest())
(OUT/'CHECK.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(timings=times,stages=stages),indent=2))
