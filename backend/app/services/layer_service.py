from geoalchemy2 import shape as ga_shape
from shapely.geometry.base import BaseGeometry

from app.core.exceptions import AppError
from app.repositories.analysis import AnalysisRepository
from app.schemas.layer import LayerCatalogueResponse, MapLayerItem

LAYER_DEFS = {
    "study_area": {
        "display_name": "Study-area boundary",
        "description": "Study-area boundary used for this analysis.",
    },
    "ml_flood": {
        "display_name": "ML-detected flood extent",
        "description": "Flood-class pixels from custom ML model after configured masking.",
    },
    "baseline_flood": {
        "display_name": "Thresholding baseline",
        "description": "Flood-class pixels from documented SAR thresholding baseline.",
    },
    "agricultural_exposure": {
        "display_name": "Potentially exposed agricultural land",
        "description": "Intersection of detected flood extent and selected agricultural land-cover classes.",
    },
    "permanent_water": {
        "display_name": "Permanent water mask",
        "description": "Permanent-water extent from JRC Global Surface Water for this run.",
    },
    "validation_reference": {
        "display_name": "Validation reference",
        "description": "Independent reference layer used for validation, when available.",
    },
}

CLASS_NAMES = {
    "study_area": "study_area",
    "ml_flood": "flood",
    "baseline_flood": "flood",
    "agricultural_exposure": "potentially_exposed_agriculture",
    "permanent_water": "permanent_water",
    "validation_reference": "reference",
}

SIMPLIFY_VERTEX_LIMIT = 15000
SIMPLIFY_TOLERANCE_DEG = 0.0005


class LayerService:
    def __init__(self, db=None, repo: AnalysisRepository | None = None):
        self.repo = repo if repo is not None else AnalysisRepository(db)

    def catalogue(self, analysis_id) -> LayerCatalogueResponse:
        run = self._run(analysis_id)
        rows = {row.layer_key: row for row in self.repo.layers_of(run.id)}
        items = []
        for layer_key, definition in LAYER_DEFS.items():
            row = rows.get(layer_key)
            available = bool(row is not None and row.available and row.geometry is not None)
            model_version = run.model_version if layer_key == "ml_flood" else None
            source_date = run.observation_end.strftime("%Y-%m-%d") if run.observation_end else None
            items.append(
                MapLayerItem(
                    layer_key=layer_key,
                    display_name=definition["display_name"],
                    available=available,
                    format="geojson",
                    area_km2=float(row.area_km2) if row is not None and row.area_km2 is not None else None,
                    feature_count=row.feature_count if row is not None else None,
                    source_date=source_date,
                    model_version=model_version,
                    description=definition["description"],
                )
            )
        return LayerCatalogueResponse(analysis_id=run.id, layers=items)

    def geojson(self, analysis_id, layer_key: str) -> dict:
        run = self._run(analysis_id)
        if layer_key not in LAYER_DEFS:
            raise AppError("LAYER_NOT_AVAILABLE", "Requested map layer is not available.", 404)
        row = self.repo.layer(run.id, layer_key)
        if row is None or not row.available or row.geometry is None:
            raise AppError("LAYER_NOT_AVAILABLE", "Requested map layer is not available.", 404)
        geom = row.geometry if isinstance(row.geometry, BaseGeometry) else ga_shape.to_shape(row.geometry)
        geom = self._simplify_if_large(geom)
        if geom is None or geom.is_empty:
            raise AppError("LAYER_NOT_AVAILABLE", "Requested map layer is not available.", 404)
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": f"{layer_key}-{row.id}",
                    "geometry": _geometry_mapping(geom),
                    "properties": {
                        "layer_key": layer_key,
                        "area_km2": float(row.area_km2) if row.area_km2 is not None else None,
                        "class_name": CLASS_NAMES[layer_key],
                    },
                }
            ],
        }

    def _simplify_if_large(self, geom: BaseGeometry) -> BaseGeometry | None:
        if geom is None or geom.is_empty:
            return None
        vertex_count = 0
        for part in getattr(geom, "geoms", [geom]):
            vertex_count += len(part.exterior.coords)
            for ring in part.interiors:
                vertex_count += len(ring.coords)
        if vertex_count <= SIMPLIFY_VERTEX_LIMIT:
            return geom
        simplified = geom.simplify(SIMPLIFY_TOLERANCE_DEG, preserve_topology=True)
        return simplified if not simplified.is_empty else geom

    def _run(self, analysis_id):
        run = self.repo.get(analysis_id)
        if run is None:
            raise AppError("ANALYSIS_NOT_FOUND", "Analysis was not found.", 404)
        return run


def _geometry_mapping(geom: BaseGeometry) -> dict:
    from shapely.geometry import mapping as shp_mapping

    return shp_mapping(geom)
