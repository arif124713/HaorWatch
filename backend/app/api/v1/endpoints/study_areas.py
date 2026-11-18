from fastapi import APIRouter, Depends

from app.api.deps import get_analysis_service, get_study_area_service
from app.schemas.analysis import AnalysisLatest
from app.schemas.study_area import StudyAreaDetail, StudyAreaListResponse
from app.services.analysis_service import AnalysisService
from app.services.study_area_service import StudyAreaService

router = APIRouter()


@router.get("/study-areas", response_model=StudyAreaListResponse)
def list_study_areas(service: StudyAreaService = Depends(get_study_area_service)):
    return service.list_areas()


@router.get("/study-areas/{slug}", response_model=StudyAreaDetail)
def study_area_detail(slug: str, service: StudyAreaService = Depends(get_study_area_service)):
    return service.get_area(slug)


@router.get("/study-areas/{slug}/analyses/latest", response_model=AnalysisLatest)
def latest_analysis(slug: str, service: AnalysisService = Depends(get_analysis_service)):
    return service.latest(slug)
