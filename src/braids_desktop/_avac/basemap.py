"""Optional visible-only online tiles, reprojected as a display mesh.

No tile is used by the scientific pipeline. Native data is never resampled.
"""
from collections import OrderedDict
import math
from pathlib import Path

from osgeo import osr
from PySide6.QtCore import QObject, QPointF, QRectF, QStandardPaths, QTimer, QUrl, Signal
from PySide6.QtGui import QImage, QPainter, QPolygonF, QTransform
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkDiskCache, QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import QGraphicsItem

from .geo import spatial_reference, has_geographic_reference
from .preferences import settings

WORLD = 20037508.342789244
PROVIDERS = {
    "osm": ("https://tile.openstreetmap.org/{z}/{x}/{y}.png", "© OpenStreetMap contributors", "https://www.openstreetmap.org/copyright"),
    "satellite": ("https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                  "Imagery © Esri, Maxar, Earthstar Geographics, GIS User Community", "https://www.arcgis.com/home/item.html?id=10df2279f9684e4a9f6a7f08febac2a9"),
}


def tile_bounds(z, x, y):
    step = 2*WORLD/(2**z)
    return -WORLD+x*step, WORLD-y*step, step


class TileItem(QGraphicsItem):
    def __init__(self, image, key, transform, origin):
        super().__init__()
        self.image = image
        z, x, y = key
        west, north, step = tile_bounds(z, x, y)
        self.mesh = []
        n = 8
        vertices = transform.TransformPoints([(west+step*c/n, north-step*r/n) for r in range(n+1) for c in range(n+1)])
        points = [QPointF(p[0]-origin[0], origin[1]-p[1]) for p in vertices]
        if any(not math.isfinite(p.x()) or not math.isfinite(p.y()) for p in points):
            raise ValueError("Background projection is undefined in this region.")
        self.bounds = QPolygonF(points).boundingRect()
        for r in range(n):
            for c in range(n):
                source = QPolygonF([QPointF(c*256/n, r*256/n), QPointF((c+1)*256/n, r*256/n),
                                    QPointF((c+1)*256/n, (r+1)*256/n), QPointF(c*256/n, (r+1)*256/n)])
                target = QPolygonF([points[r*(n+1)+c], points[r*(n+1)+c+1],
                                    points[(r+1)*(n+1)+c+1], points[(r+1)*(n+1)+c]])
                matrix = QTransform()
                if QTransform.quadToQuad(source, target, matrix):
                    self.mesh.append((QRectF(c*256/n, r*256/n, 256/n, 256/n), matrix))
        self.setZValue(-20)

    def boundingRect(self):
        return self.bounds

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        for source, matrix in self.mesh:
            painter.save()
            painter.setTransform(matrix, True)
            # Subpixel overlap hides sampling seams between adjacent mesh cells.
            source = source.adjusted(-.25, -.25, .25, .25)
            painter.drawImage(source, self.image, source)
            painter.restore()


class Basemap(QObject):
    message = Signal(str)

    def __init__(self, view):
        super().__init__(view)
        self.view = view
        self.provider = "none"
        self.items = {}
        self.images = OrderedDict()
        self.pending = {}
        self.queue = []
        self.wanted = set()
        self.generation = 0
        self.failed = set()
        self.network = QNetworkAccessManager(self)
        cache = QNetworkDiskCache(self)
        directory = Path(QStandardPaths.writableLocation(QStandardPaths.CacheLocation))/"map-tiles"
        cache.setCacheDirectory(str(directory))
        cache.setMaximumCacheSize(256*1024*1024)
        self.network.setCache(cache)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self.refresh)
        view.extent_changed.connect(lambda _: self.schedule())

    def schedule(self):
        if self.provider != "none":
            self.timer.start()

    def clear(self):
        self.timer.stop()
        self.generation += 1
        for reply in list(self.pending):
            reply.abort()
        self.pending.clear()
        self.queue.clear()
        self.wanted.clear()
        self.failed.clear()
        for item in self.items.values():
            if item.scene():
                self.view.scene().removeItem(item)
        self.items.clear()

    def set_provider(self, provider):
        self.clear()
        self.provider = provider if provider in PROVIDERS else "none"
        if self.provider != "none" and self.view.terrain and not has_geographic_reference(self.view.terrain):
            self.provider = "none"
            self.message.emit("Background disabled: the terrain has no geographic CRS.")
        self.view.attribution.setText("")
        if self.provider != "none":
            _, text, url = PROVIDERS[self.provider]
            self.view.attribution.setText(f'<a style="color:#dfeefa" href="{url}">{text}</a>')
            self.view.attribution.setToolTip("Online background: visible map tiles are requested from the provider. Terrain and releases are not uploaded.")
        self.view._position_controls()
        self.schedule()

    def refresh(self):
        if self.provider == "none":
            return
        if not self.view.terrain:
            self.message.emit("Load a georeferenced terrain before displaying an online background.")
            return
        if not has_geographic_reference(self.view.terrain):
            self.set_provider("none")
            self.message.emit("Background disabled: the terrain has no geographic CRS.")
            return
        try:
            self._refresh()
        except (ValueError, RuntimeError, OverflowError, KeyError, AttributeError) as exc:
            self.message.emit(f"Background unavailable: {exc}")

    def _refresh(self):
        terrain = self.view.terrain
        target = spatial_reference(terrain.wkt or terrain.raster.crs_authid)
        mercator = spatial_reference("EPSG:3857")
        forward = osr.CoordinateTransformation(target, mercator)
        self.inverse = osr.CoordinateTransformation(mercator, target)
        m = terrain.raster.metadata
        self.origin = float(m["xmin"]), float(m["ymax"])
        rect = self.view.mapToScene(self.view.viewport().rect()).boundingRect()
        samples = [self.view.from_scene(QPointF(rect.left()+rect.width()*c/2, rect.top()+rect.height()*r/2))
                   for r in range(3) for c in range(3)]
        points = forward.TransformPoints(samples)
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        if not all(math.isfinite(v) for v in xs+ys):
            raise ValueError("the current extent is outside the background CRS")
        left, right = max(-WORLD, min(xs)), min(WORLD, max(xs))
        bottom, top = max(-WORLD, min(ys)), min(WORLD, max(ys))
        if left >= right or bottom >= top:
            return
        resolution = max((right-left)/max(1, self.view.viewport().width()), 1e-6)
        zoom = min(18, max(0, round(math.log2(2*WORLD/(256*resolution)))))
        while True:
            n = 2**zoom
            x0, x1 = max(0, int((left+WORLD)/(2*WORLD)*n)), min(n-1, int((right+WORLD)/(2*WORLD)*n))
            y0, y1 = max(0, int((WORLD-top)/(2*WORLD)*n)), min(n-1, int((WORLD-bottom)/(2*WORLD)*n))
            keys = {(zoom, x, y) for x in range(x0, x1+1) for y in range(y0, y1+1)}
            if len(keys) <= 32 or zoom == 0:
                break
            zoom -= 1
        self.wanted = keys
        for key in list(self.items):
            if key not in keys:
                self.view.scene().removeItem(self.items.pop(key))
        active = set(self.pending.values())
        self.queue = []
        for key in sorted(keys):
            if key in self.items:
                continue
            identity = (self.provider, *key)
            if identity in self.images:
                self._show(key, self.images[identity])
            elif key not in active and identity not in self.failed:
                self.queue.append(key)
        self._pump()

    def _show(self, key, image):
        item = TileItem(image, key, self.inverse, self.origin)
        self.view.scene().addItem(item)
        self.items[key] = item

    def _pump(self):
        while self.queue and len(self.pending) < 4:
            key = self.queue.pop(0)
            if key not in self.wanted:
                continue
            z, x, y = key
            url = settings().value(f"tiles/{self.provider}", PROVIDERS[self.provider][0]).format(z=z, x=x, y=y)
            request = QNetworkRequest(QUrl(url))
            request.setRawHeader(b"User-Agent", b"AVACDesktop/0.1 (+https://github.com/cgotelli/AVAC-Desktop)")
            request.setTransferTimeout(15000)
            request.setAttribute(QNetworkRequest.CacheLoadControlAttribute, QNetworkRequest.PreferCache)
            reply = self.network.get(request)
            self.pending[reply] = key
            generation, provider = self.generation, self.provider
            reply.finished.connect(lambda r=reply, k=key, g=generation, p=provider: self._finished(r, k, g, p))

    def _finished(self, reply, key, generation, provider):
        self.pending.pop(reply, None)
        try:
            if generation != self.generation:
                return
            image = QImage.fromData(reply.readAll()) if reply.error() == QNetworkReply.NoError else QImage()
            if image.isNull():
                self.failed.add((provider, *key))
                self.message.emit("Some background tiles are unavailable. Check the connection or choose another background.")
            else:
                image = image.scaled(256, 256)
                identity = (provider, *key)
                self.images[identity] = image
                self.images.move_to_end(identity)
                while len(self.images) > 64:
                    self.images.popitem(last=False)
                if key in self.wanted and self.view.terrain:
                    self._show(key, image)
        except (ValueError, RuntimeError) as exc:
            self.message.emit(f"Background projection unavailable: {exc}")
        finally:
            reply.deleteLater()
            if generation == self.generation:
                self._pump()
