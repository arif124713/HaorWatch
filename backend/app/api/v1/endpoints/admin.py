import uuid

from fastapi import APIRouter, Depends

from app.api.deps import get_analysis_service, require_operator_key
from app.schemas.analysis import TriggerAccepted, TriggerAnalysisRequest
from app.services.analysis_service import AnalysisService

router = APIRouter()


@router.post("/admin/analyses", response_model=TriggerAccepted, status_code=202)
def trigger_analysis(
    request: TriggerAnalysisRequest,
    operator_key: str = Depends(require_operator_key),
    service: AnalysisService = Depends(get_analysis_service),
):
    return service.trigger_analysis(request)
