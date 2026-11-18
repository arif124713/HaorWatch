import uuid

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_analysis_service
from app.schemas.analysis import AnalysisDetail, AnalysisListResponse
from app.services.analysis_service import AnalysisService

router = APIRouter()


@router.get("/study-areas/{slug}/analyses", response_model=AnalysisListResponse)
def list_analyses(
    slug: str,
    limit: int = Query(default=10, ge=1, le=100),
    service: AnalysisService = Depends(get_analysis_service),
):
    return service.list(slug, limit)


@router.get("/analyses/{analysis_id}", response_model=AnalysisDetail)
def analysis_detail(analysis_id: uuid.UUID, service: AnalysisService = Depends(get_analysis_service)):
    return service.detail(analysis_id)
