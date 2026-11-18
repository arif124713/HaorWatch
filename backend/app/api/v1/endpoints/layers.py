import json
import uuid

from fastapi import APIRouter, Depends, Response

from app.api.deps import get_layer_service
from app.schemas.layer import LayerCatalogueResponse
from app.services.layer_service import LayerService

router = APIRouter()


@router.get("/analyses/{analysis_id}/layers", response_model=LayerCatalogueResponse)
def layer_catalogue(analysis_id: uuid.UUID, service: LayerService = Depends(get_layer_service)):
    return service.catalogue(analysis_id)


@router.get("/analyses/{analysis_id}/layers/{layer_key}")
def layer_geojson(
    analysis_id: uuid.UUID, layer_key: str, service: LayerService = Depends(get_layer_service)
):
    feature_collection = service.geojson(analysis_id, layer_key)
    return Response(
        content=json.dumps(feature_collection),
        media_type="application/geo+json",
    )
