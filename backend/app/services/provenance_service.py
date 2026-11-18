from app.repositories.analysis import AnalysisRepository
from app.schemas.analysis import SourceOut, SourcesResponse

ROLE_BY_KEY = {
    "sentinel1_sar": "Flood detection input imagery",
    "sen1floods11": "Training and evaluation dataset",
    "jrc_gsw": "Permanent-water exclusion mask",
    "esa_worldcover": "Agricultural land-cover overlay",
    "sentinel2_reference": "Optional cloud-free reference proxy",
    "copernicus_ems": "Optional independent flood reference",
}


class ProvenanceService:
    def __init__(self, db=None, repo: AnalysisRepository | None = None):
        self.repo = repo if repo is not None else AnalysisRepository(db)

    def sources(self, analysis_id) -> SourcesResponse:
        run = self.repo.get(analysis_id)
        if run is None:
            from app.core.exceptions import AppError

            raise AppError("ANALYSIS_NOT_FOUND", "Analysis was not found.", 404)

        config_sources = (run.processing_config or {}).get("sources")
        if config_sources:
            sources = [SourceOut(**item) for item in config_sources]
            return SourcesResponse(analysis_id=run.id, sources=sources)

        sources = []
        for asset in self.repo.dataset_assets_for(run.study_area_id):
            sources.append(
                SourceOut(
                    dataset_key=asset.dataset_key,
                    display_name=asset.display_name,
                    provider=asset.provider,
                    role=ROLE_BY_KEY.get(asset.dataset_key),
                    version=asset.source_version,
                    acquired_at=asset.acquired_at,
                    period_start=asset.period_start,
                    period_end=asset.period_end,
                    resolution_m=float(asset.resolution_m) if asset.resolution_m is not None else None,
                    crs=asset.crs,
                    source_url=asset.source_url,
                    status="used" if asset.storage_uri else "reference_only",
                    notes=asset.license_note,
                )
            )
        return SourcesResponse(analysis_id=run.id, sources=sources)
