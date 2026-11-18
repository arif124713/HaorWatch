from geoalchemy2 import shape as ga_shape
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from app.core.exceptions import AppError
from app.repositories.analysis import AnalysisRepository
from app.repositories.study_area import StudyAreaRepository
from app.schemas.study_area import StudyAreaDetail, StudyAreaItem, StudyAreaListResponse


class StudyAreaService:
    def __init__(self, db=None, repo: StudyAreaRepository | None = None):
        self.repo = repo if repo is not None else StudyAreaRepository(db)

    def list_areas(self) -> StudyAreaListResponse:
        items = []
        for area in self.repo.list_active():
            items.append(self._to_item(area, with_bounds=True))
        return StudyAreaListResponse(items=items)

    def get_area(self, slug: str) -> StudyAreaDetail:
        area = self.repo.get_by_slug(slug)
        if area is None:
            raise AppError("STUDY_AREA_NOT_FOUND", "Study area was not found.", 404)
        return self._to_detail(area)

    def _to_item(self, area, with_bounds: bool = False) -> StudyAreaItem:
        return StudyAreaItem(
            id=area.id,
            slug=area.slug,
            name=area.name,
            division=area.division,
            districts=area.districts,
            country_code=area.country_code,
            area_km2=float(area.area_km2) if area.area_km2 is not None else None,
            bounds=self.repo.bounds_for(area) if with_bounds else None,
            is_active=area.is_active,
        )

    def _boundary_geojson(self, area) -> dict | None:
        if area.boundary is None:
            return None
        geom = area.boundary if isinstance(area.boundary, BaseGeometry) else ga_shape.to_shape(area.boundary)
        feature = {
            "type": "Feature",
            "properties": {"slug": area.slug, "name": area.name},
            "geometry": mapping(geom),
        }
        return {"type": "FeatureCollection", "features": [feature]}

    def _to_detail(self, area) -> StudyAreaDetail:
        return StudyAreaDetail(
            id=area.id,
            slug=area.slug,
            name=area.name,
            division=area.division,
            districts=area.districts,
            country_code=area.country_code,
            area_km2=float(area.area_km2) if area.area_km2 is not None else None,
            bounds=self.repo.bounds_for(area),
            is_active=area.is_active,
            boundary_geojson=self._boundary_geojson(area),
            metadata=area.metadata_json,
        )
