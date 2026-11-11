"""Georeferenced native Qt viewport with zoom-dependent terrain detail."""
from __future__ import annotations

from .i18n import tr, trf, tr_list, number, parse_number
import math
from copy import deepcopy
import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap, QTransform
from PySide6.QtWidgets import (QApplication, QFrame, QGraphicsItem, QGraphicsPathItem, QGraphicsPixmapItem, QHBoxLayout, QLabel,
                               QGraphicsScene, QGraphicsView, QToolButton, QVBoxLayout)

from .geo import Rings, Terrain, probe
from .raster_style import PALETTES
from .theme import LEGEND_BLUE


def bin_icon():
    """Draw a crisp, font-independent bin in the map controls' foreground colour."""
    pixmap = QPixmap(48, 48)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.scale(2, 2)
    painter.setPen(QPen(QColor("#f2f7fb"), 1.7, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    painter.drawRoundedRect(QRectF(6, 7, 12, 14), 1.5, 1.5)
    painter.drawRoundedRect(QRectF(9, 3, 6, 3), 1, 1)
    painter.drawLine(QPointF(4, 6), QPointF(20, 6))
    painter.drawLine(QPointF(10, 11), QPointF(10, 17))
    painter.drawLine(QPointF(14, 11), QPointF(14, 17))
    painter.end()
    return QIcon(pixmap)


def colour_image(values: np.ndarray, *, terrain: bool = False, cell: float = 1.0, diverging: bool = False,
                 limits: tuple[float, float] | None = None, palette: str | None = None,
                 zero_transparent: bool = False, hidden_mask=None) -> tuple[QImage, float, float]:
    """Generate display colors only. The scientific source array is untouched."""
    valid = np.isfinite(values)
    low, high = limits or (float(np.nanmin(values)), float(np.nanmax(values)))
    if diverging and palette is None:
        absolute = max(abs(low), abs(high))
        low, high = -absolute, absolute
    if diverging and high == low:
        t = np.full(values.shape, .5)
    else:
        t = np.clip((np.nan_to_num(values, nan=low) - low) / (high-low if high != low else 1), 0, 1)
    if palette in PALETTES:
        stops = np.asarray([[QColor(c).red(), QColor(c).green(), QColor(c).blue()] for c in PALETTES[palette]])
    elif terrain:
        stops = np.array([[53, 78, 73], [107, 126, 100], [162, 158, 122], [185, 180, 157], [228, 232, 227]])
    elif diverging:
        stops = np.array([[39, 103, 181], [216, 225, 228], [231, 118, 60]])
    else:
        stops = np.array([[30, 95, 168], [22, 185, 207], [131, 220, 151], [250, 208, 80], [240, 94, 54]])
    knots = np.linspace(0, 1, len(stops))
    rgb = np.stack([np.interp(t, knots, stops[:, i]) for i in range(3)], axis=-1)
    if terrain and min(values.shape) >= 2:
        filled = np.where(valid, values, low)
        spacing=cell if isinstance(cell,tuple) else (cell,cell)
        gy, gx = np.gradient(filled, *spacing)
        # Display hillshade, unit vertical scale, light from the northwest.
        # Rows are south-to-north. Dot the upward surface normal (-gx,-gy,1)
        # with a northwestern light (-.5,+.5,.707), before the display Y flip.
        illumination = (gx*.5 - gy*.5 + .707) / np.sqrt(gx*gx + gy*gy + 1)
        rgb *= np.clip(.55 + .6*illumination[..., None], .25, 1.12)
    rgba = np.empty((*values.shape, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    rgba[..., 3] = (valid & ((values != 0) if zero_transparent else True)).astype(np.uint8)*255
    if hidden_mask is not None:
        rgba[..., 3][hidden_mask] = 0
    north = np.ascontiguousarray(rgba[::-1])
    return QImage(north.data, north.shape[1], north.shape[0], north.strides[0], QImage.Format_RGBA8888).copy(), low, high


def terrain_display(terrain: Terrain) -> tuple[QImage, float, float]:
    """Static thumbnail helper; the main map renders native/overview tiles."""
    raster = terrain.raster
    stride = max(1, math.ceil(max(raster.z.shape)/1600))
    # Explicit display overview; never used as a solver input.
    return colour_image(raster.z[::stride, ::stride], terrain=True,
                        cell=float(raster.metadata["cellsize"])*stride,
                        limits=(float(np.nanmin(raster.z)),float(np.nanmax(raster.z))))


class MapView(QGraphicsView):
    coordinate = Signal(str)
    extent_changed = Signal(object)
    profile_drawn = Signal(object)
    gauge_placed = Signal(float, float)
    lake_seed = Signal(float, float)
    geometry_changed = Signal(object, object)
    tool_changed = Signal(str)
    styles_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        # Cartography is anchored to the viewport, so scene scroll/resize
        # optimizations must not reuse pixels at its previous screen position.
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setBackgroundBrush(QColor("#0c141e"))
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setMouseTracking(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.terrain: Terrain | None = None
        self.terrain_item: QGraphicsItem | None = None
        self._terrain_range=None
        self.release_item: QGraphicsPathItem | None = None
        self.field_item: QGraphicsPixmapItem | None = None
        self.field_raster = None
        self.field_hide_rings = []
        self._field_mask = None
        self.additional_fields = {}
        self.additional_items = {}
        self.lake_item = None
        self.mesh_items = []
        self.mesh_snapshot = None
        self.decorations = dict(legends=True, scale=True, north=True)
        self.time_text = ""
        self.tool = "pan"
        self.points: list[list[float]] = []
        self.line_item = None
        self.profile_item = None
        self.analysis_items = []
        self.analysis_visibility = dict(profile=True, gauge=True)
        self.low, self.high = 0., 1.
        self.field_label = "Terrain elevation [m]"
        self.fit_pending = False
        self.domain_rect = QRectF()
        self.rings, self.sketch_lines = [], []
        self.sketch_item = None
        self.handles = []
        self.selection = self.drag_vertex = None
        self.styles = {}
        self.model_tools = True
        self.terrain_low, self.terrain_high = 0., 1.
        self._build_controls()
        self.attribution = QLabel(tr(self.viewport()))
        self.attribution.setOpenExternalLinks(True)
        self.attribution.setStyleSheet(f"background: {LEGEND_BLUE}; color: #f0f7fc; padding: 4px; font-size: 10px;")
        from .basemap import Basemap
        self.basemap = Basemap(self)
        from .preferences import appearance
        self.apply_appearance(QApplication.instance().property("appearance") or appearance())

    def _build_controls(self):
        self.map_controls = QFrame(self.viewport())
        self.navigation_controls = QFrame(self.viewport())
        layout = QVBoxLayout(self.map_controls)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        navigation, geometry = QVBoxLayout(self.navigation_controls), layout
        navigation.setContentsMargins(0, 0, 0, 0)
        navigation.setSpacing(3)
        layout.setAlignment(Qt.AlignTop)
        self.tool_buttons = {}
        for key, symbol, hint, callback in (
            ("zoom_in", "+", "Zoom in", lambda: self.zoom(1.3)),
            ("zoom_out", "−", "Zoom out", lambda: self.zoom(1/1.3)),
            ("pan", "✥", "Pan map", lambda: self.set_tool("pan")),
            ("polygon", "⬠", "Draw release polygon · click vertices, double-click or Enter to finish", lambda: self.set_tool("polygon")),
            ("line", "╱", "Draw sketch line · click vertices, double-click or Enter to finish", lambda: self.set_tool("line")),
            ("edit", "✎", "Edit geometry · select a polygon or line, then drag its vertices", lambda: self.set_tool("edit")),
            ("delete", "", "Delete selected polygon or sketch line", self.delete_selection),
            ("finish", "✓", "Finish drawing (Enter)", self.finish_drawing),
            ("cancel", "×", "Cancel drawing / clear selection (Escape)", lambda: self.set_tool("pan"))):
            button = QToolButton()
            button.setObjectName("MapControl")
            button.setText(tr(symbol))
            if key == "delete":
                button.setIcon(bin_icon())
                button.setIconSize(QSize(24, 24))
            button.setFixedSize(36, 36)
            button.setToolTip(tr(hint))
            button.setAccessibleName(tr(hint))
            if key in ("pan", "polygon", "line", "edit"):
                button.setCheckable(True)
            button.clicked.connect(callback)
            self.tool_buttons[key] = button
            (navigation if key in ("zoom_in", "zoom_out", "pan") else geometry).addWidget(button)
        self.style_button = QToolButton(self.navigation_controls)
        self.style_button.setText(tr("◐"))
        self.style_button.setObjectName("MapControl")
        self.style_button.setFixedSize(36, 36)
        self.style_button.setToolTip(tr("Layer appearance: palettes, colorbar limits and opacity"))
        self.style_button.clicked.connect(self.edit_raster_style)
        navigation.addWidget(self.style_button)
        self.set_tool("pan")

    def _position_controls(self):
        self.navigation_controls.adjustSize()
        self.map_controls.adjustSize()
        self.navigation_controls.move(max(0, self.viewport().width()-self.navigation_controls.width()-12),
                                      max(12, (self.viewport().height()-self.navigation_controls.height())//2))
        self.map_controls.move(12, 12)
        self.navigation_controls.show()
        self.navigation_controls.raise_()
        self.map_controls.setVisible(self.model_tools)
        self.map_controls.raise_()
        if hasattr(self, "attribution"):
            self.attribution.setVisible(bool(self.attribution.text()))
            self.attribution.adjustSize()
            self.attribution.setMaximumWidth(max(1, self.viewport().width()-24))
            self.attribution.move(max(0, self.viewport().width()-self.attribution.width()-12), 8)
            self.attribution.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "navigation_controls"):
            self._position_controls()
        if hasattr(self, "basemap"):
            self.basemap.schedule()

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        # QGraphicsView also scrolls viewport-child widgets; restore overlay
        # positions after every camera movement, not just after resizing.
        if hasattr(self, "navigation_controls"):
            self._position_controls()

    def apply_appearance(self, mode):
        self.setBackgroundBrush(QColor("#e2e9ef" if mode == "light" else "#0c141e"))

    def edit_raster_style(self):
        from .raster_style import RasterStyleDialog
        if self.terrain:
            RasterStyleDialog(self).exec()

    def appearance_layers(self):
        layers=[('Terrain','terrain')]
        if self.field_raster is not None:layers.append((self.field_label,self.field_label))
        layers.extend((label,label) for label in self.additional_fields)
        return layers

    def appearance_palettes(self,key):
        return list(PALETTES)

    def refresh_appearance(self,key):self.refresh_styles()

    def default_palette(self, key):
        return "Terrain" if key == "terrain" else ("Blue–red" if "displacement" in key.lower() else "Ocean to fire")

    def default_zero_transparent(self, key):
        # DEM sea-level/no-data collars are frequently encoded as exactly
        # zero.  Terrain follows the same transparent-zero default as result
        # fields; users may still override it in raster appearance settings.
        return True

    def range_for(self, key):
        if key=='terrain' and self._terrain_range is not None:return self._terrain_range
        raster = self.terrain.raster if key == "terrain" else self.additional_fields.get(key, self.field_raster)
        low, high = float(np.nanmin(raster.z)), float(np.nanmax(raster.z))
        if key != "terrain" and "displacement" in key.lower():
            limit = max(abs(low), abs(high))
            return -limit, limit
        return low, high

    def _style_image(self, raster, key):
        stride = max(1, math.ceil(max(raster.z.shape)/1600))
        style = self.styles.get(key, {})
        limits = style.get("limits") or self.range_for(key)
        hidden_mask = None
        if key == self.field_label and self.field_hide_rings:
            grid = (raster.z.shape, float(raster.x[0]), float(raster.x[-1]), float(raster.y[0]), float(raster.y[-1]))
            if self._field_mask is None or self._field_mask[0] != grid:
                from .geometry import display_mask_from_rings as release_mask_from_rings
                # Clip only the bounded display overview, never scientific
                # arrays or exports. Polygon holes retain land/island snow.
                self._field_mask = grid, release_mask_from_rings(self.field_hide_rings, raster.x[::stride], raster.y[::stride])
            hidden_mask = self._field_mask[1]
        return colour_image(raster.z[::stride, ::stride], terrain=key == "terrain",
            cell=float(raster.metadata["cellsize"])*stride, limits=limits,
            palette=style.get("palette", self.default_palette(key)),
            zero_transparent=style.get("zero_transparent", self.default_zero_transparent(key)), hidden_mask=hidden_mask)

    def refresh_styles(self):
        if self.terrain:
            style=self.styles.get('terrain',{})
            self.terrain_low,self.terrain_high=style.get('limits') or self.range_for('terrain')
            self.terrain_item.configure((self.terrain_low,self.terrain_high),
                palette=style.get('palette',self.default_palette('terrain')),
                zero_transparent=style.get('zero_transparent',self.default_zero_transparent('terrain')))
            self.terrain_item.setOpacity(style.get('opacity',1.))
            if self.field_raster is not None:
                self.set_field(self.field_raster, self.field_label, hide_rings=self.field_hide_rings)
            else:
                self.low, self.high = self.terrain_low, self.terrain_high
            for label, raster in list(self.additional_fields.items()):
                self.set_additional_field(raster, label)
            self.viewport().update()

    def set_model_tools(self, enabled):
        self.model_tools = enabled
        for key in ("polygon", "line", "edit", "delete"):
            self.tool_buttons[key].setVisible(enabled)
        if not enabled and self.tool in ("polygon", "line", "edit"):
            self.set_tool("pan")
        self._drawing_actions()
        self._position_controls()

    def clear(self):
        self.set_tool("pan")
        if hasattr(self, "basemap"):
            self.basemap.clear()
        self.scene().clear()
        self.mesh_items = []
        self.mesh_snapshot = None
        self.additional_fields, self.additional_items = {}, {}
        self.lake_item = None
        self.analysis_items = []
        self.terrain = self.terrain_item = self.release_item = self.field_item = None
        self._terrain_range=None
        self.field_raster = self.line_item = self.sketch_item = None
        self.field_hide_rings, self._field_mask = [], None
        self.profile_item = None
        self.field_label = "Terrain elevation [m]"
        self.low, self.high = 0., 1.
        self.rings, self.sketch_lines, self.handles = [], [], []
        self.domain_rect = QRectF()
        self.resetTransform()
        self.setSceneRect(QRectF(-10000, -10000, 20000, 20000))
        self.centerOn(0, 0)
        self.viewport().update()
        self._position_controls()

    def set_terrain(self, terrain: Terrain, rings: Rings, image=None, low=None, high=None) -> None:
        self.clear()
        if low is None:low=float(np.nanmin(terrain.raster.z))
        if high is None:high=float(np.nanmax(terrain.raster.z))
        self.terrain, self.low, self.high = terrain, low, high
        self._terrain_range=(low,high)
        self.terrain_low, self.terrain_high = low, high
        self.field_item = self.line_item = None
        self.points.clear()
        from .terrain_tiles import TerrainTiles
        self.terrain_item=TerrainTiles(terrain.raster,(low,high),palette=self.default_palette('terrain'))
        self.scene().addItem(self.terrain_item)
        meta = terrain.raster.metadata
        width, height = float(meta["xmax"])-float(meta["xmin"]), float(meta["ymax"])-float(meta["ymin"])
        self.domain_rect = QRectF(0, 0, width, height)
        self.setSceneRect(self.domain_rect.adjusted(-width*20, -height*20, width*20, height*20))
        self.set_release(rings)
        self.fit_domain()
        self.refresh_styles()
        self.basemap.schedule()

    def set_release(self, rings: Rings) -> None:
        self.rings = deepcopy(rings)
        if self.release_item is not None:
            self.scene().removeItem(self.release_item)
        path = QPainterPath()
        path.setFillRule(Qt.OddEvenFill)
        for exterior, holes in rings:
            # Separate union paths retain overlap coverage; holes use odd-even.
            polygon = QPainterPath()
            polygon.setFillRule(Qt.OddEvenFill)
            for ring in (exterior, *holes):
                polygon.moveTo(self.to_scene(*ring[0]))
                for point in ring[1:]:
                    polygon.lineTo(self.to_scene(*point))
                polygon.closeSubpath()
            path = path.united(polygon)
        pen = QPen(QColor("#ffb264"), 1.5)
        pen.setCosmetic(True)
        self.release_item = self.scene().addPath(path, pen, QColor(246, 143, 52, 100))
        self.release_item.setZValue(3)

    def set_lake(self, rings):
        if self.lake_item is not None:
            self.scene().removeItem(self.lake_item)
        self.lake_item = None
        if not self.terrain or not rings:
            return
        path = QPainterPath()
        path.setFillRule(Qt.OddEvenFill)
        for exterior, holes in rings:
            for ring in (exterior, *holes):
                path.moveTo(self.to_scene(*ring[0]))
                for point in ring[1:]:
                    path.lineTo(self.to_scene(*point))
                path.closeSubpath()
        pen = QPen(QColor("#3d9fc7"), 1.6)
        pen.setCosmetic(True)
        self.lake_item = self.scene().addPath(path, pen, QColor(57, 163, 205, 65))
        self.lake_item.setZValue(2.5)

    def set_mesh(self, snapshot):
        from .mesh_view import MeshItem
        self.mesh_snapshot = snapshot
        if not self.terrain:
            return
        origin = (float(self.terrain.raster.metadata['xmin']), float(self.terrain.raster.metadata['ymax']))
        layers = snapshot.layers if snapshot is not None else ()
        if len(layers) != len(self.mesh_items) or any(item.origin != origin or item.layer.name != layer.name
                for item, layer in zip(self.mesh_items, layers)):
            for item in self.mesh_items:
                self.scene().removeItem(item)
            self.mesh_items = []
            for layer in layers:
                item = MeshItem(layer, origin)
                self.scene().addItem(tr(item))
                self.mesh_items.append(item)
        else:
            for item, layer in zip(self.mesh_items, layers):
                item.set_layer(layer)
        self.viewport().update()

    def set_additional_field(self, raster, label):
        self.additional_fields[label] = raster
        if label in self.additional_items:
            self.scene().removeItem(self.additional_items[label])
        image, _, _ = self._style_image(raster, label)
        item = self.scene().addPixmap(QPixmap.fromImage(image))
        m = raster.metadata
        item.setPos(self.to_scene(float(m["xmin"]), float(m["ymax"])))
        item.setTransform(QTransform.fromScale((float(m["xmax"])-float(m["xmin"]))/image.width(),
                                             (float(m["ymax"])-float(m["ymin"]))/image.height()))
        item.setOpacity(self.styles.get(label, {}).get("opacity", 1.))
        item.setZValue(2)
        self.additional_items[label] = item
        self.viewport().update()

    def clear_additional_fields(self):
        for item in self.additional_items.values():
            self.scene().removeItem(item)
        self.additional_fields, self.additional_items = {}, {}

    def export_image(self):
        """Render scene and cartography without interactive tool widgets."""
        image = QImage(self.viewport().size(), QImage.Format_ARGB32_Premultiplied)
        image.fill(Qt.transparent)
        painter = QPainter(image)
        self.render(painter, QRectF(image.rect()), self.viewport().rect())
        if self.attribution.text():
            from PySide6.QtGui import QTextDocument
            document = QTextDocument()
            document.setHtml(self.attribution.text())
            credits = document.toPlainText()
            metrics = painter.fontMetrics()
            width = metrics.horizontalAdvance(credits)+16
            painter.fillRect(QRectF(8, 8, width, metrics.height()+8), QColor(LEGEND_BLUE))
            painter.setPen(QColor("white"))
            painter.drawText(16, 12+metrics.ascent(), tr(credits))
        painter.end()
        return image

    def to_scene(self, x: float, y: float) -> QPointF:
        m = self.terrain.raster.metadata
        return QPointF(x-float(m["xmin"]), float(m["ymax"])-y)

    def from_scene(self, point: QPointF) -> tuple[float, float]:
        m = self.terrain.raster.metadata
        return point.x()+float(m["xmin"]), float(m["ymax"])-point.y()

    def fit_domain(self) -> None:
        if self.terrain:
            self.fitInView(self.domain_rect, Qt.KeepAspectRatio)
            self._expand_canvas()
            self._save_extent()

    def restore_extent(self, extent: list[float]) -> None:
        if len(extent) != 4 or not all(math.isfinite(v) for v in extent) or extent[2] <= 0 or extent[3] <= 0:
            raise ValueError("Invalid saved map extent.")
        self.fitInView(QRectF(*extent), Qt.KeepAspectRatio)
        self._expand_canvas()

    def _expand_canvas(self):
        # QGraphicsView uses internal scroll values for its camera even when
        # scrollbars are hidden. Grow before the camera approaches an edge.
        visible = self.mapToScene(self.viewport().rect()).boundingRect()
        margin = max(visible.width(), visible.height(), 1)*20
        target = visible.adjusted(-margin, -margin, margin, margin)
        if not self.sceneRect().contains(target):
            self.setSceneRect(self.sceneRect().united(target))

    def _save_extent(self) -> None:
        rect = self.mapToScene(self.viewport().rect()).boundingRect()
        self.extent_changed.emit([rect.x(), rect.y(), rect.width(), rect.height()])

    def set_field(self, raster, label: str, *, hide_rings=None) -> None:
        if hide_rings is not self.field_hide_rings:
            self.field_hide_rings, self._field_mask = hide_rings or [], None
        self.field_raster, self.field_label = raster, label
        if self.field_item is not None:
            self.scene().removeItem(self.field_item)
            self.field_item = None
        if not np.any(np.isfinite(raster.z)):
            raise ValueError("This field contains no finite values.")
        stride = max(1, math.ceil(max(raster.z.shape)/1600))
        image, self.low, self.high = self._style_image(raster, label)
        item = self.scene().addPixmap(QPixmap.fromImage(image))
        m = raster.metadata
        item.setPos(self.to_scene(float(m["xmin"]), float(m["ymax"])))
        item.setTransform(QTransform.fromScale((float(m["xmax"])-float(m["xmin"]))/image.width(),
                                             (float(m["ymax"])-float(m["ymin"]))/image.height()))
        item.setZValue(1)
        item.setOpacity(self.styles.get(label, {}).get("opacity", 1.))
        self.field_item = item
        self.viewport().update()

    def clear_field(self) -> None:
        self.clear_additional_fields()
        if self.field_item:
            self.scene().removeItem(self.field_item)
        self.field_item = self.field_raster = None
        self.field_hide_rings, self._field_mask = [], None
        self.field_label = "Terrain elevation [m]"
        self.low, self.high = self.terrain_low, self.terrain_high
        self.viewport().update()

    def set_tool(self, tool: str) -> None:
        self.tool = tool
        self.points.clear()
        self.drag_vertex = self.selection = None
        self._clear_handles()
        if self.line_item is not None:
            self.scene().removeItem(self.line_item)
            self.line_item = None
        self.setDragMode(QGraphicsView.ScrollHandDrag if tool == "pan" else QGraphicsView.NoDrag)
        self.setCursor(Qt.OpenHandCursor if tool == "pan" else Qt.CrossCursor)
        for key in ("pan", "polygon", "line", "edit"):
            self.tool_buttons[key].setChecked(key == tool)
        self.tool_changed.emit(tool)
        self._drawing_actions()

    def _drawing_actions(self):
        for key in ("finish", "cancel"):
            self.tool_buttons[key].setVisible(self.model_tools and self.tool in ("polygon", "line"))
        self._position_controls()

    def zoom(self, factor):
        scale = self.transform().m11()*factor
        if 1e-6 < scale < 10000:
            self.scale(factor, factor)
            self._expand_canvas()
            self._save_extent()

    def wheelEvent(self, event) -> None:
        factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2
        self.zoom(factor)

    def mouseMoveEvent(self, event) -> None:
        if self.drag_vertex is not None:
            contour, index = self.drag_vertex
            coords = self._selected_contours()[contour]
            x, y = self.from_scene(self.mapToScene(event.position().toPoint()))
            coords[index] = [x, y]
            if self.selection[0] == "polygon" and index == 0:
                coords[-1] = [x, y]
            self._redraw_geometry()
            event.accept()
            return
        if self.terrain:
            x, y = self.from_scene(self.mapToScene(event.position().toPoint()))
            hit = probe(self.field_raster or self.terrain.raster, x, y)
            text = trf('E {0:,.2f}  N {1:,.2f}  |  {2}', x, y, self.terrain.raster.crs_authid or tr('Unreferenced XY [m]'))
            if hit:
                row, col, value = hit
                text += trf('  |  cell ({0}, {1})  ·  Elevation: {2:.8g} m', row, col, value) if np.isfinite(value) else "  |  NoData"
            self.coordinate.emit(text)
        super().mouseMoveEvent(event)
        if self.tool == "pan":
            self._expand_canvas()

    def mousePressEvent(self, event) -> None:
        if self.terrain and event.button() == Qt.LeftButton and self.tool != "pan":
            x, y = self.from_scene(self.mapToScene(event.position().toPoint()))
            if self.tool == "gauge":
                self.gauge_placed.emit(x, y)
            elif self.tool == "lake":
                self.lake_seed.emit(x, y)
            elif self.tool == "edit":
                self._select_or_drag(event.position().toPoint())
            elif self.tool in ("profile", "polygon", "line"):
                if not self.points or self.points[-1] != [x, y]:
                    self.points.append([x, y])
                self._draw_draft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if self.tool in ("profile", "polygon", "line"):
            self.finish_drawing()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self.drag_vertex is not None:
            self.drag_vertex = None
            self.geometry_changed.emit(deepcopy(self.rings), deepcopy(self.sketch_lines))
            event.accept()
            return
        super().mouseReleaseEvent(event)
        self._save_extent()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.set_tool("pan")
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.finish_drawing()
        elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self.tool == "edit":
            self.delete_selection()
        else:
            super().keyPressEvent(event)

    def _draw_draft(self):
        if self.line_item is not None:
            self.scene().removeItem(self.line_item)
        path = QPainterPath(self.to_scene(*self.points[0]))
        for point in self.points[1:]:
            path.lineTo(self.to_scene(*point))
        if self.tool == "polygon" and len(self.points) > 2:
            path.closeSubpath()
        pen = QPen(QColor("#55dcff"), 2)
        pen.setCosmetic(True)
        self.line_item = self.scene().addPath(path, pen)
        self.line_item.setZValue(5)

    def finish_drawing(self):
        minimum = 3 if self.tool == "polygon" else 2
        if len(self.points) < minimum:
            return
        points, tool = deepcopy(self.points), self.tool
        if tool == "profile":
            if self.profile_item is not None:
                self.scene().removeItem(self.profile_item)
            self.profile_item, self.line_item = self.line_item, None
            self.profile_drawn.emit(points)
        elif tool == "polygon":
            self.rings.append((np.asarray(points+[points[0]], dtype=float), []))
            self.geometry_changed.emit(deepcopy(self.rings), deepcopy(self.sketch_lines))
        elif tool == "line":
            self.sketch_lines.append(points)
            self.geometry_changed.emit(deepcopy(self.rings), deepcopy(self.sketch_lines))
        self.set_tool("pan")

    def set_sketch_lines(self, lines):
        self.sketch_lines = deepcopy(lines)
        if self.sketch_item is not None:
            self.scene().removeItem(self.sketch_item)
        path = QPainterPath()
        if self.terrain:
            for line in lines:
                path.moveTo(self.to_scene(*line[0]))
                for point in line[1:]:
                    path.lineTo(self.to_scene(*point))
        pen = QPen(QColor("#55dcff"), 2)
        pen.setCosmetic(True)
        self.sketch_item = self.scene().addPath(path, pen)
        self.sketch_item.setZValue(4)

    def set_analysis_visibility(self, kind, visible):
        self.analysis_visibility[kind] = visible
        for item in self.analysis_items:
            if item.data(0) == kind:
                item.setVisible(visible)

    def set_analysis_features(self, features):
        for item in self.analysis_items:
            if item.scene() is self.scene():
                self.scene().removeItem(item)
        self.analysis_items = []
        if not self.terrain:
            return
        if self.profile_item is not None:
            self.scene().removeItem(self.profile_item)
            self.profile_item = None
        pen = QPen(QColor("#159ec8"), 2)
        pen.setCosmetic(True)
        for feature in features:
            points = feature["coords"]
            path = QPainterPath()
            if feature["kind"] == "profile":
                path.moveTo(self.to_scene(*points[0]))
                for point in points[1:]:
                    path.lineTo(self.to_scene(*point))
                item = self.scene().addPath(path, pen)
            else:
                # A fixed-pixel cross remains visible at any zoom level.
                path.moveTo(-6, 0)
                path.lineTo(6, 0)
                path.moveTo(0, -6)
                path.lineTo(0, 6)
                item = self.scene().addPath(path, pen)
                item.setFlag(item.GraphicsItemFlag.ItemIgnoresTransformations)
                item.setPos(self.to_scene(*points[0]))
            item.setZValue(6)
            item.setToolTip(tr(feature["name"]))
            item.setData(0, feature["kind"])
            item.setVisible(self.analysis_visibility.get(feature["kind"], True))
            self.analysis_items.append(item)

    def _selected_contours(self):
        kind, index = self.selection
        if kind == "line":
            return [self.sketch_lines[index]]
        exterior, holes = self.rings[index]
        return [exterior, *holes]

    def _clear_handles(self):
        for item in self.handles:
            self.scene().removeItem(item)
        self.handles = []

    def _redraw_geometry(self):
        self.set_release(self.rings)
        self.set_sketch_lines(self.sketch_lines)
        self._clear_handles()
        if self.selection:
            for contour in self._selected_contours():
                points = contour[:-1] if self.selection[0] == "polygon" else contour
                for point in points:
                    item = self.scene().addEllipse(-5, -5, 10, 10, QPen(QColor("#101722")), QColor("#77d9ff"))
                    item.setPos(self.to_scene(*point))
                    item.setFlag(QGraphicsItem.ItemIgnoresTransformations)
                    item.setZValue(7)
                    self.handles.append(item)

    def _select_or_drag(self, position):
        if self.selection and self.tool == "edit":
            for c, contour in enumerate(self._selected_contours()):
                points = contour[:-1] if self.selection[0] == "polygon" else contour
                for i, point in enumerate(points):
                    pixel = self.mapFromScene(self.to_scene(*point))
                    if (pixel-position).manhattanLength() <= 12:
                        self.drag_vertex = c, i
                        return
        self.selection = None
        point = self.mapToScene(position)
        # Prefer a nearby line over a polygon below it.
        from PySide6.QtGui import QPainterPathStroker
        stroker = QPainterPathStroker()
        stroker.setWidth(16/max(self.transform().m11(), 1e-20))
        for i, line in enumerate(self.sketch_lines):
            path = QPainterPath(self.to_scene(*line[0]))
            for vertex in line[1:]:
                path.lineTo(self.to_scene(*vertex))
            if stroker.createStroke(path).contains(point):
                self.selection = "line", i
                break
        if self.selection is None:
            for i, (exterior, holes) in enumerate(self.rings):
                path = QPainterPath()
                path.setFillRule(Qt.OddEvenFill)
                for ring in (exterior, *holes):
                    path.moveTo(self.to_scene(*ring[0]))
                    for vertex in ring[1:]:
                        path.lineTo(self.to_scene(*vertex))
                    path.closeSubpath()
                if path.contains(point) or stroker.createStroke(path).contains(point):
                    self.selection = "polygon", i
                    break
        self._redraw_geometry()

    def delete_selection(self):
        if self.selection:
            kind, index = self.selection
            (self.rings if kind == "polygon" else self.sketch_lines).pop(index)
            self.selection = None
            self._clear_handles()
            self.geometry_changed.emit(deepcopy(self.rings), deepcopy(self.sketch_lines))

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        super().drawForeground(painter, rect)
        if not self.terrain:
            return
        painter.save()
        painter.resetTransform()
        dpr = self.viewport().devicePixelRatioF()
        # QPainter's resetTransform preserves devicePixelRatio; use logical px.
        w, h = self.viewport().width(), self.viewport().height()
        if self.decorations.get("north", True):
            painter.save()
            painter.translate(self.navigation_controls.geometry().center().x()-44*.7/2, 12)
            if self.attribution.text():
                painter.translate(0, self.attribution.height()+8)
            painter.scale(.7, .7)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(LEGEND_BLUE))
            painter.drawRoundedRect(QRectF(0, 0, 44, 76), 6, 6)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(QRectF(0, 3, 44, 22), Qt.AlignCenter, tr("N"))
            needle = QPainterPath(QPointF(22, 26))
            needle.lineTo(13, 61)
            needle.lineTo(22, 54)
            needle.closeSubpath()
            painter.fillPath(needle, QColor("#ffffff"))
            needle = QPainterPath(QPointF(22, 26))
            needle.lineTo(31, 61)
            needle.lineTo(22, 54)
            needle.closeSubpath()
            painter.fillPath(needle, QColor("#80d7f1"))
            painter.restore()
        if self.time_text and self.decorations.get('time',True):
            metrics = painter.fontMetrics()
            width = metrics.horizontalAdvance(self.time_text)+20
            painter.setBrush(QColor(LEGEND_BLUE))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(QRectF(w/2-width/2, 12, width, 28), 5, 5)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(QRectF(w/2-width/2, 12, width, 28), Qt.AlignCenter, tr(self.time_text))
        from PySide6.QtGui import QLinearGradient
        legends = [("terrain", "Terrain elevation [m]", self.terrain_low, self.terrain_high)] if self.terrain_item is not None and self.terrain_item.isVisible() else []
        if self.field_item is not None:
            legends.append((self.field_label, self.field_label, self.low, self.high))
        for label in self.additional_fields:
            low, high = self.range_for(label)
            limits = self.styles.get(label, {}).get("limits")
            legends.append((label, label, *(limits or (low, high))))
        legends = [entry for entry in legends if self.decorations.get("legends", True)
                   and self.styles.get(entry[0], {}).get("legend_visible", True)]
        metrics = painter.fontMetrics()
        padding, gap, ramp_height = 8, 4, 6
        panel_height = padding*2 + metrics.height()*2 + gap*2 + ramp_height
        for index, (key, title, low, high) in enumerate(legends):
            title = tr(title)
            low_text, high_text = trf('{0:.4g}', low), trf('{0:.4g}', high)
            panel_width = max(174, metrics.horizontalAdvance(title)+padding*2,
                              metrics.horizontalAdvance(low_text)+metrics.horizontalAdvance(high_text)+padding*2+24)
            bottom = h-16-index*(panel_height+8)
            left, top = 14, bottom-panel_height
            content_left = left+padding
            content_width = panel_width-padding*2
            title_baseline = top+padding+metrics.ascent()
            ramp_top = top+padding+metrics.height()+gap
            values_baseline = ramp_top+ramp_height+gap+metrics.ascent()
            backing = QColor(LEGEND_BLUE)
            backing.setAlpha(245)
            painter.setPen(Qt.NoPen)
            painter.setBrush(backing)
            painter.drawRoundedRect(QRectF(left, top, panel_width, panel_height), 5, 5)
            painter.setPen(QColor("#dfeefa"))
            painter.drawText(content_left, title_baseline, tr(title))
            gradient = QLinearGradient(content_left, 0, content_left+content_width, 0)
            colors = PALETTES.get(self.styles.get(key, {}).get("palette", self.default_palette(key)), PALETTES["Terrain"])
            for i, color in enumerate(colors):
                gradient.setColorAt(i/(len(colors)-1), QColor(color))
            painter.fillRect(QRectF(content_left, ramp_top, content_width, ramp_height), gradient)
            painter.drawText(content_left, values_baseline, tr(low_text))
            painter.drawText(content_left+content_width-metrics.horizontalAdvance(high_text), values_baseline, tr(high_text))
        if not self.decorations.get("scale", True):
            painter.restore()
            return
        metres_per_pixel = 1 / max(self.transform().m11(), 1e-20)
        raw = metres_per_pixel*100
        power = 10**math.floor(math.log10(raw))
        length = max(1, math.floor(raw/power))*power
        pixels = length/metres_per_pixel
        scale_text = trf('{0:g} m', length)
        scale_width = max(pixels, metrics.horizontalAdvance(scale_text))
        scale_left = w-16-scale_width-padding*2
        scale_line_y = h-16-padding
        scale_baseline = scale_line_y-gap-metrics.descent()
        scale_top = scale_baseline-metrics.ascent()-padding
        backing = QColor(LEGEND_BLUE)
        backing.setAlpha(245)
        painter.setPen(Qt.NoPen)
        painter.setBrush(backing)
        painter.drawRoundedRect(QRectF(scale_left, scale_top, scale_width+padding*2, h-16-scale_top), 5, 5)
        painter.setPen(QPen(QColor("#d9e7f1"), 2))
        painter.drawLine(QPointF(scale_left+padding, scale_line_y), QPointF(scale_left+padding+pixels, scale_line_y))
        painter.drawText(QPointF(scale_left+padding, scale_baseline), tr(scale_text))
        painter.restore()
