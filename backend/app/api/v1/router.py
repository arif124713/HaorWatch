from fastapi import APIRouter

from app.api.v1.endpoints import admin, analyses, health, layers, metrics, reports, study_areas

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(study_areas.router, tags=["study-areas"])
api_router.include_router(analyses.router, tags=["analyses"])
api_router.include_router(layers.router, tags=["layers"])
api_router.include_router(metrics.router, tags=["metrics"])
api_router.include_router(reports.router, tags=["reports"])
api_router.include_router(admin.router, tags=["admin"])
