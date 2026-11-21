from datetime import date, datetime
import uuid
from decimal import Decimal

from pydantic import BaseModel, Field


class StudyAreaRef(BaseModel):
    slug: str
    name: str


class ModelInfo(BaseModel):
    name: str | None = None
    version: str | None = None


class BaselineInfo(BaseModel):
    method: str | None = None
    display_name: str | None = None


class PermanentWaterMaskInfo(BaseModel):
    applied: bool = False
    source: str | None = None
    threshold: float | None = None
    note: str | None = None


class ValidationInfo(BaseModel):
    status: str = "not_available"
    note: str | None = None


class AnalysisSummaryOut(BaseModel):
    study_area_km2: float | None = Field(default=None, ge=0)
    detected_flood_area_km2: float | None = Field(default=None, ge=0)
    baseline_flood_area_km2: float | None = Field(default=None, ge=0)
    mapped_agricultural_area_km2: float | None = Field(default=None, ge=0)
    potentially_exposed_agricultural_area_km2: float | None = Field(default=None, ge=0)
    potentially_exposed_agricultural_share_percent: float | None = Field(default=None, ge=0, le=100)
    valid_pixel_count: int | None = Field(default=None, ge=0)
    nodata_pixel_count: int | None = Field(default=None, ge=0)


class AnalysisLatest(BaseModel):
    id: uuid.UUID
    study_area: StudyAreaRef
    status: str
    analysis_label: str
    observation_start: datetime | None = None
    observation_end: datetime | None = None
    created_at: datetime
    completed_at: datetime | None = None
    progress_percent: int | None = Field(default=None, ge=0, le=100)
    progress_message: str | None = None
    is_stale: bool = False
    model: ModelInfo
    baseline: BaselineInfo
    permanent_water_mask: PermanentWaterMaskInfo
    validation: ValidationInfo
    summary: AnalysisSummaryOut
    available_layers: list[str] = []


class AnalysisDetail(AnalysisLatest):
    processing_config: dict | None = None
    error_code: str | None = None
    error_message_safe: str | None = None
    created_by: str | None = None


class AnalysisListItem(BaseModel):
    id: uuid.UUID
    status: str
    analysis_label: str
    observation_start: datetime | None = None
    observation_end: datetime | None = None
    created_at: datetime
    completed_at: datetime | None = None


class AnalysisListResponse(BaseModel):
    items: list[AnalysisListItem]
    total: int


class ConfusionMatrixOut(BaseModel):
    true_positive: int | None = Field(default=None, ge=0)
    false_positive: int | None = Field(default=None, ge=0)
    true_negative: int | None = Field(default=None, ge=0)
    false_negative: int | None = Field(default=None, ge=0)


class MethodMetricsOut(BaseModel):
    method_key: str
    method_name: str | None = None
    precision: float | None = Field(default=None, ge=0, le=1)
    recall: float | None = Field(default=None, ge=0, le=1)
    f1: float | None = Field(default=None, ge=0, le=1)
    iou: float | None = Field(default=None, ge=0, le=1)
    confusion_matrix: ConfusionMatrixOut
    evaluated_at: datetime | None = None
    notes: str | None = None


class TestDatasetOut(BaseModel):
    name: str | None = None
    version: str | None = None
    split_strategy: str | None = None
    test_pixel_count: int | None = Field(default=None, ge=0)


class MetricsResponse(BaseModel):
    analysis_id: uuid.UUID
    validation_status: str
    positive_class: str = "flood"
    test_dataset: TestDatasetOut | None = None
    methods: list[MethodMetricsOut]
    limitations: list[str] = []


class SourceOut(BaseModel):
    dataset_key: str
    display_name: str | None = None
    provider: str | None = None
    role: str | None = None
    version: str | None = None
    acquired_at: datetime | None = None
    period_start: date | None = None
    period_end: date | None = None
    resolution_m: float | None = None
    crs: str | None = None
    source_url: str | None = None
    status: str = "used"
    notes: str | None = None


class SourcesResponse(BaseModel):
    analysis_id: uuid.UUID
    sources: list[SourceOut]


class TriggerAnalysisRequest(BaseModel):
    study_area_slug: str
    observation_start: datetime | None = None
    observation_end: datetime | None = None
    prepared_dir: str | None = None
    run_ml: bool = True
    run_baseline: bool = True
    run_exposure: bool = True


class TriggerAccepted(BaseModel):
    analysis_id: uuid.UUID
    status: str
    message: str


def dec_to_float(value: Decimal | float | None) -> float | None:
    if value is None:
        return None
    return float(value)
