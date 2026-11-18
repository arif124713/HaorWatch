from datetime import datetime, timezone

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.repositories.analysis import AnalysisRepository, utcnow
from app.repositories.study_area import StudyAreaRepository
from app.schemas.analysis import (
    AnalysisDetail,
    AnalysisLatest,
    AnalysisListItem,
    AnalysisListResponse,
    AnalysisSummaryOut,
    BaselineInfo,
    ConfusionMatrixOut,
    MetricsResponse,
    MethodMetricsOut,
    ModelInfo,
    PermanentWaterMaskInfo,
    StudyAreaRef,
    TestDatasetOut,
    TriggerAccepted,
    TriggerAnalysisRequest,
    ValidationInfo,
    dec_to_float,
)

BASELINE_DISPLAY = {
    "otsu_thresholding": "Otsu SAR Thresholding",
    "change_detection": "SAR Change Detection",
}

LIMITATIONS = [
    "Public training labels may not fully represent haor conditions.",
    "Permanent and seasonal water may be confused.",
    "SAR can struggle with flooded vegetation.",
    "Results are potential exposure estimates, not confirmed crop damage.",
]


class AnalysisService:
    def __init__(self, db=None, repo: AnalysisRepository | None = None, area_repo: StudyAreaRepository | None = None):
        self.repo = repo if repo is not None else AnalysisRepository(db)
        self.area_repo = area_repo if area_repo is not None else StudyAreaRepository(db)

    def latest(self, slug: str) -> AnalysisLatest:
        area = self._area(slug)
        run = self.repo.latest_for_study_area(area.id)
        if run is None:
            raise AppError("ANALYSIS_NOT_FOUND", "No analysis is available for this study area.", 404)
        return self._assemble(run, area)

    def detail(self, analysis_id) -> AnalysisDetail:
        run = self._run(analysis_id)
        area = self._area_by_id(run.study_area_id)
        base = self._assemble(run, area)
        return AnalysisDetail(
            **base.model_dump(),
            processing_config=run.processing_config,
            error_code=run.error_code,
            error_message_safe=run.error_message_safe,
            created_by=run.created_by,
        )

    def list(self, slug: str, limit: int = 10) -> AnalysisListResponse:
        area = self._area(slug)
        runs, total = self.repo.list_for_study_area(area.id, limit)
        items = [
            AnalysisListItem(
                id=run.id,
                status=run.status,
                analysis_label=run.analysis_label,
                observation_start=run.observation_start,
                observation_end=run.observation_end,
                created_at=run.created_at,
                completed_at=run.completed_at,
            )
            for run in runs
        ]
        return AnalysisListResponse(items=items, total=total)

    def metrics(self, analysis_id) -> MetricsResponse:
        run = self._run(analysis_id)
        rows = self.repo.metrics_of(run.id)
        methods = []
        test_dataset = None
        for row in rows:
            asset = None
            if row.test_dataset_asset_id is not None:
                asset = self.repo.dataset_asset(row.test_dataset_asset_id)
            methods.append(
                MethodMetricsOut(
                    method_key=row.method_key,
                    method_name=row.method_name,
                    precision=dec_to_float(row.precision_score),
                    recall=dec_to_float(row.recall_score),
                    f1=dec_to_float(row.f1_score),
                    iou=dec_to_float(row.iou_score),
                    confusion_matrix=ConfusionMatrixOut(
                        true_positive=row.true_positive,
                        false_positive=row.false_positive,
                        true_negative=row.true_negative,
                        false_negative=row.false_negative,
                    ),
                    evaluated_at=row.evaluated_at,
                    notes=row.notes,
                )
            )
            if test_dataset is None:
                test_dataset = TestDatasetOut(
                    name=asset.display_name if asset else None,
                    version=asset.source_version if asset else None,
                    split_strategy=row.split_strategy,
                    test_pixel_count=row.test_pixel_count,
                )
        limitations = list(LIMITATIONS)
        if run.validation_status == "not_available":
            limitations.append("No independent validation reference was available for this run.")
        return MetricsResponse(
            analysis_id=run.id,
            validation_status=run.validation_status,
            test_dataset=test_dataset,
            methods=methods,
            limitations=limitations,
        )

    def trigger_analysis(self, request: TriggerAnalysisRequest) -> TriggerAccepted:
        area = self.area_repo.get_by_slug(request.study_area_slug)
        if area is None:
            raise AppError("STUDY_AREA_NOT_FOUND", "Study area was not found.", 404)
        label = "Operator requested analysis"
        if request.observation_start and request.observation_end:
            label = f"Requested analysis {request.observation_start:%Y-%m-%d} to {request.observation_end:%Y-%m-%d}"
        run = self.repo.create_run(
            study_area_id=area.id,
            status="queued",
            analysis_label=label,
            observation_start=request.observation_start,
            observation_end=request.observation_end,
            created_by="operator_api",
            processing_config={
                "requested": request.model_dump(mode="json"),
                "note": "Pipeline must be executed via scripts.run_analysis",
            },
        )
        return TriggerAccepted(
            analysis_id=run.id,
            status=run.status,
            message="Analysis request accepted.",
        )

    def _area(self, slug: str):
        area = self.area_repo.get_by_slug(slug)
        if area is None:
            raise AppError("STUDY_AREA_NOT_FOUND", "Study area was not found.", 404)
        return area

    def _area_by_id(self, area_id):
        for area in self.area_repo.list_active():
            if area.id == area_id:
                return area
        raise AppError("STUDY_AREA_NOT_FOUND", "Study area was not found.", 404)

    def _run(self, analysis_id):
        run = self.repo.get(analysis_id)
        if run is None:
            raise AppError("ANALYSIS_NOT_FOUND", "Analysis was not found.", 404)
        return run

    def _assemble(self, run, area) -> AnalysisLatest:
        summary_row = self.repo.summary_of(run.id)
        layers = self.repo.layers_of(run.id)
        available_layers = [layer.layer_key for layer in layers if layer.available]

        config = run.processing_config or {}
        mask_config = config.get("permanent_water", {})
        threshold = run.permanent_water_threshold
        threshold_value = float(threshold) if threshold is not None else mask_config.get("threshold")

        return AnalysisLatest(
            id=run.id,
            study_area=StudyAreaRef(slug=area.slug, name=area.name),
            status=run.status,
            analysis_label=run.analysis_label,
            observation_start=run.observation_start,
            observation_end=run.observation_end,
            created_at=run.created_at,
            completed_at=run.completed_at,
            progress_percent=run.progress_percent,
            progress_message=run.progress_message,
            is_stale=self._is_stale(run),
            model=ModelInfo(
                name=run.model_name or "HaorWatch Flood Classifier",
                version=run.model_version,
            ),
            baseline=BaselineInfo(
                method=run.baseline_method,
                display_name=BASELINE_DISPLAY.get(run.baseline_method or "", run.baseline_method),
            ),
            permanent_water_mask=PermanentWaterMaskInfo(
                applied=bool(run.permanent_water_mask_applied),
                source=mask_config.get("source") if run.permanent_water_mask_applied else None,
                threshold=threshold_value,
                note=mask_config.get("note"),
            ),
            validation=ValidationInfo(status=run.validation_status, note=run.validation_note),
            summary=self._summary(summary_row),
            available_layers=available_layers,
        )

    def _summary(self, summary_row) -> AnalysisSummaryOut:
        if summary_row is None:
            return AnalysisSummaryOut()
        return AnalysisSummaryOut(
            study_area_km2=dec_to_float(summary_row.study_area_km2),
            detected_flood_area_km2=dec_to_float(summary_row.detected_flood_area_km2),
            baseline_flood_area_km2=dec_to_float(summary_row.baseline_flood_area_km2),
            mapped_agricultural_area_km2=dec_to_float(summary_row.mapped_agricultural_area_km2),
            potentially_exposed_agricultural_area_km2=dec_to_float(
                summary_row.potentially_exposed_agricultural_area_km2
            ),
            potentially_exposed_agricultural_share_percent=dec_to_float(
                summary_row.potentially_exposed_agricultural_share_percent
            ),
            valid_pixel_count=summary_row.valid_pixel_count,
            nodata_pixel_count=summary_row.nodata_pixel_count,
        )

    @staticmethod
    def _is_stale(run) -> bool:
        settings = get_settings()
        reference = run.completed_at or run.observation_end or run.created_at
        if reference is None:
            return False
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - reference).days
        return age_days > settings.stale_after_days
