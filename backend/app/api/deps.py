import secrets

from fastapi import Depends, Header
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.database import get_db
from app.core.exceptions import AppError
from app.services.analysis_service import AnalysisService
from app.services.layer_service import LayerService
from app.services.provenance_service import ProvenanceService
from app.services.report_service import ReportService
from app.services.study_area_service import StudyAreaService


def get_settings_dep() -> Settings:
    return get_settings()


def get_study_area_service(db: Session = Depends(get_db)) -> StudyAreaService:
    return StudyAreaService(db)


def get_analysis_service(db: Session = Depends(get_db)) -> AnalysisService:
    return AnalysisService(db)


def get_layer_service(db: Session = Depends(get_db)) -> LayerService:
    return LayerService(db)


def get_report_service(db: Session = Depends(get_db)) -> ReportService:
    return ReportService(db)


def get_provenance_service(db: Session = Depends(get_db)) -> ProvenanceService:
    return ProvenanceService(db)


def get_analysis_runner():
    """Return the callable that executes a queued analysis in a subprocess.

    Exposed as a dependency so tests and deployments can substitute a runner
    without changing the endpoint.
    """
    from app.services.analysis_worker import execute_run_subprocess

    return execute_run_subprocess


def require_operator_key(
    x_operator_key: str | None = Header(default=None, alias="X-Operator-Key"),
    settings: Settings = Depends(get_settings_dep),
) -> str:
    if not x_operator_key or not secrets.compare_digest(x_operator_key, settings.operator_api_key):
        raise AppError("INVALID_OPERATOR_KEY", "Missing or invalid operator key.", 401)
    return x_operator_key
