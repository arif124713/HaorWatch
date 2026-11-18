from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from app.api.deps import get_settings_dep
from app.core.config import Settings
from app.schemas.common import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings_dep)):
    return HealthResponse(
        status="ok",
        service="haorwatch-api",
        version=settings.app_version,
        timestamp=datetime.now(timezone.utc),
    )
