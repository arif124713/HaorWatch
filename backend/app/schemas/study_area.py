import uuid

from pydantic import BaseModel


class StudyAreaItem(BaseModel):
    id: uuid.UUID
    slug: str
    name: str
    division: str | None = None
    districts: list[str] | None = None
    country_code: str = "BD"
    area_km2: float | None = None
    bounds: list[float] | None = None
    is_active: bool = True


class StudyAreaDetail(StudyAreaItem):
    boundary_geojson: dict | None = None
    metadata: dict | None = None


class StudyAreaListResponse(BaseModel):
    items: list[StudyAreaItem]
