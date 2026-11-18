import uuid

from fastapi import APIRouter, Depends

from app.api.deps import get_report_service
from app.schemas.report import ReportResponse
from app.services.report_service import ReportService

router = APIRouter()


@router.get("/analyses/{analysis_id}/reports/{language}", response_model=ReportResponse)
def analysis_report(
    analysis_id: uuid.UUID, language: str, service: ReportService = Depends(get_report_service)
):
    return service.get_report(analysis_id, language)
