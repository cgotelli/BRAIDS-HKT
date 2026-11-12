"""Editable QGIS domain and atomic replacement-style boundary assignment."""
from dataclasses import asdict

from qgis.PyQt.QtCore import QVariant
from qgis.PyQt import sip
from qgis.core import (QgsCoordinateReferenceSystem, QgsFeature, QgsField, QgsGeometry,
    QgsLineSymbol, QgsPointXY, QgsProject, QgsVectorLayer)

from braids_core.boundaries.domain import PolygonDomain, BoundaryAssignment
from ..core.edges import reassign_boundary_faces


def domain_from_layer(layer):
    if layer is None or not layer.isValid():
        raise ValueError("Select a valid calculation-domain polygon")
    features = list(layer.getSelectedFeatures()) or list(layer.getFeatures())
    if len(features) != 1:
        raise ValueError("Select exactly one domain polygon")
    geometry = features[0].geometry()
    if geometry.isEmpty() or not geometry.isGeosValid():
        raise ValueError("The calculation domain must be a valid nonempty polygon")
    if geometry.isMultipart():
        raise ValueError("The calculation domain must be one polygon")
    rings = geometry.asPolygon()
    if len(rings) != 1 or len(rings[0]) < 4:
        raise ValueError("The current preparation route requires one polygon without holes")
    vertices = [(p.x(), p.y()) for p in rings[0]]
    if vertices[0] == vertices[-1]:
        vertices.pop()
    return PolygonDomain(str(layer.customProperty("braids/domain_id", "domain")), vertices, layer.crs().toWkt())


def layer_from_domain(domain, name="BRAIDS calculation domain"):
    layer = QgsVectorLayer("Polygon", name, "memory")
    layer.setCustomProperty("braids/domain_id", domain.domain_id)
    layer.setCrs(QgsCoordinateReferenceSystem(domain.crs_wkt))
    feature = QgsFeature()
    feature.setGeometry(QgsGeometry.fromPolygonXY([[QgsPointXY(*xy) for xy in domain.vertices_xy]]))
    layer.dataProvider().addFeatures([feature])
    layer.updateExtents()
    QgsProject.instance().addMapLayer(layer)
    return layer


class BoundaryEdges:
    def __init__(self):
        self.layer = None
        self.domain = None

    def create(self, domain, assignments=()):
        if len({a.edge_id for a in assignments}) != len(assignments):
            raise ValueError("The boundary-face editor does not yet support multiple partial assignments on one edge")
        # Remove only the old generated layer from this controller. Historical
        # datasets remain on disk and are never used as current assignments.
        if self.layer is not None and not sip.isdeleted(self.layer) and QgsProject.instance().mapLayer(self.layer.id()):
            QgsProject.instance().removeMapLayer(self.layer.id())
        self.domain = domain
        self.layer = QgsVectorLayer("LineString", "BRAIDS boundary faces", "memory")
        self.layer.setCustomProperty('braids/input_role','boundary_faces')
        self.layer.setCrs(QgsCoordinateReferenceSystem(domain.crs_wkt))
        provider = self.layer.dataProvider()
        provider.addAttributes([QgsField(k, QVariant.String) for k in ("edge_id", "bc_id", "kind")]+[
            QgsField("start_m", QVariant.Double), QgsField("end_m", QVariant.Double)])
        self.layer.updateFields()
        rows = {a.edge_id: a for a in assignments}
        features = []
        for edge in domain.edges:
            f = QgsFeature(self.layer.fields())
            f.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(*edge.start_xy), QgsPointXY(*edge.end_xy)]))
            a = rows.get(edge.edge_id)
            f.setAttributes([edge.edge_id, a.bc_id if a else "", a.kind if a else "wall",
                a.start_m if a else 0., a.end_m if a and a.end_m is not None else edge.length_m])
            features.append(f)
        provider.addFeatures(features)
        self.layer.updateExtents()
        self.layer.renderer().setSymbol(QgsLineSymbol.createSimple({"line_color": "#2369b2", "line_width": ".7"}))
        QgsProject.instance().addMapLayer(self.layer)
        return self.layer

    def assignments(self, domain):
        if self.layer is None or sip.isdeleted(self.layer) or not QgsProject.instance().mapLayer(self.layer.id()):
            raise ValueError("Create boundary faces from the current domain")
        if (domain.domain_id != self.domain.domain_id or tuple(domain.vertices_xy) != tuple(self.domain.vertices_xy)
            or QgsCoordinateReferenceSystem(domain.crs_wkt) != QgsCoordinateReferenceSystem(self.domain.crs_wkt)):
            raise ValueError("The domain changed. Recreate its boundary faces before preparing.")
        return [BoundaryAssignment(str(f["edge_id"]), str(f["bc_id"]), str(f["kind"]),
                                   float(f["start_m"]), float(f["end_m"]))
                for f in self.layer.getFeatures() if f["kind"] != "wall"]

    def clear(self, domain, name):
        """Remove an assignment, not its geometry; freed faces become walls."""
        current = self.assignments(domain)
        if not name or not any(a.bc_id == name for a in current):
            raise ValueError("Choose an existing boundary name to clear")
        if self.layer.isEditable():
            raise ValueError("Finish editing boundary attributes first")
        changes = {f.id(): {1: "", 2: "wall", 3: 0., 4: f.geometry().length()}
                   for f in self.layer.getFeatures() if str(f["bc_id"]) == name}
        if not self.layer.dataProvider().changeAttributeValues(changes):
            raise ValueError("Could not clear the boundary assignment")
        self.layer.triggerRepaint()

    def apply(self, domain, kind, name):
        current = self.assignments(domain)
        selected = [str(f["edge_id"]) for f in self.layer.getSelectedFeatures()]
        candidate = reassign_boundary_faces(domain, current, selected, kind, name)
        rows = {a.edge_id: a for a in candidate}
        changes = {}
        lengths = {e.edge_id: e.length_m for e in domain.edges}
        for f in self.layer.getFeatures():
            a = rows.get(str(f["edge_id"]))
            changes[f.id()] = {1: a.bc_id if a else "", 2: a.kind if a else "wall",
                3: a.start_m if a else 0., 4: a.end_m if a and a.end_m is not None else lengths[str(f["edge_id"])]}
        if self.layer.isEditable():
            raise ValueError("Finish editing boundary attributes before applying assignments")
        if not self.layer.dataProvider().changeAttributeValues(changes):
            raise ValueError("Could not save boundary assignments")
        self.layer.triggerRepaint()
        return candidate
