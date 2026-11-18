import uuid

from pydantic import BaseModel


class MapLayerItem(BaseModel):
    layer_key: str
    display_name: str
    available: bool = False
    format: str = "geojson"
    area_km2: float | None = None
    feature_count: int | None = None
    source_date: str | None = None
    model_version: str | None = None
    description: str


class LayerCatalogueResponse(BaseModel):
    analysis_id: uuid.UUID
    layers: list[MapLayerItem]
