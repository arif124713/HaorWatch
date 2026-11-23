from fastapi import APIRouter, BackgroundTasks, Depends

from app.api.deps import get_analysis_runner, get_analysis_service, require_operator_key
from app.schemas.analysis import TriggerAccepted, TriggerAnalysisRequest
from app.services.analysis_service import AnalysisService

router = APIRouter()


@router.post("/admin/analyses", response_model=TriggerAccepted, status_code=202)
def trigger_analysis(
    request: TriggerAnalysisRequest,
    background_tasks: BackgroundTasks,
    operator_key: str = Depends(require_operator_key),
    service: AnalysisService = Depends(get_analysis_service),
    runner=Depends(get_analysis_runner),
):
    # AnalysisService validates the observation window (or prepared_dir) before a
    # run is ever created, so an accepted run always has a worker to execute it.
    accepted = service.trigger_analysis(request)
    background_tasks.add_task(
        runner,
        run_id=str(accepted.analysis_id),
        slug=request.study_area_slug,
        start=request.observation_start.isoformat() if request.observation_start else None,
        end=request.observation_end.isoformat() if request.observation_end else None,
        prepared_dir=request.prepared_dir,
    )
    return accepted

