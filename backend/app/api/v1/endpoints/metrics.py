import uuid

from fastapi import APIRouter, Depends

from app.api.deps import get_analysis_service, get_provenance_service
from app.schemas.analysis import MetricsResponse, SourcesResponse
from app.services.analysis_service import AnalysisService
from app.services.provenance_service import ProvenanceService

router = APIRouter()


@router.get("/analyses/{analysis_id}/metrics", response_model=MetricsResponse)
def analysis_metrics(analysis_id: uuid.UUID, service: AnalysisService = Depends(get_analysis_service)):
    return service.metrics(analysis_id)


@router.get("/analyses/{analysis_id}/sources", response_model=SourcesResponse)
def analysis_sources(analysis_id: uuid.UUID, service: ProvenanceService = Depends(get_provenance_service)):
    return service.sources(analysis_id)
