from fastapi import APIRouter, BackgroundTasks, Depends

from app.api.deps import get_analysis_service, require_operator_key
from app.schemas.analysis import TriggerAccepted, TriggerAnalysisRequest
from app.services.analysis_service import AnalysisService
from app.services.analysis_worker import execute_run_subprocess

router = APIRouter()


@router.post("/admin/analyses", response_model=TriggerAccepted, status_code=202)
def trigger_analysis(
    request: TriggerAnalysisRequest,
    background_tasks: BackgroundTasks,
    operator_key: str = Depends(require_operator_key),
    service: AnalysisService = Depends(get_analysis_service),
):
    accepted = service.trigger_analysis(request)
    if request.observation_start is not None and request.observation_end is not None:
        background_tasks.add_task(
            execute_run_subprocess,
            run_id=str(accepted.analysis_id),
            slug=request.study_area_slug,
            start=request.observation_start.isoformat(),
            end=request.observation_end.isoformat(),
        )
    return accepted
