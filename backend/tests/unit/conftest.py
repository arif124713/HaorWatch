import uuid
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.deps import (
    get_analysis_runner,
    get_analysis_service,
    get_layer_service,
    get_provenance_service,
    get_report_service,
    get_study_area_service,
)
from app.core.exceptions import AppError
from app.main import create_app
from app.schemas.analysis import (
    AnalysisLatest,
    AnalysisSummaryOut,
    BaselineInfo,
    MetricsResponse,
    ModelInfo,
    PermanentWaterMaskInfo,
    StudyAreaRef,
    ValidationInfo,
)
from app.schemas.layer import LayerCatalogueResponse, MapLayerItem
from app.schemas.report import ReportResponse
from app.schemas.study_area import StudyAreaDetail, StudyAreaItem, StudyAreaListResponse


def make_run(**overrides):
    fields = dict(
        id=uuid.uuid4(),
        slug="hakaluki-haor",
        name="Hakaluki Haor",
        status="completed",
        analysis_label="Test analysis",
        created_at=datetime(2026, 5, 20, tzinfo=timezone.utc),
        completed_at=datetime(2026, 5, 21, tzinfo=timezone.utc),
        observation_start=datetime(2026, 5, 1, tzinfo=timezone.utc),
        observation_end=datetime(2026, 5, 13, tzinfo=timezone.utc),
        progress_percent=100,
        progress_message=None,
        model_name="HaorWatch Flood Classifier",
        model_version="0.1.0",
        baseline_method="otsu_thresholding",
        permanent_water_mask_applied=True,
        permanent_water_threshold=Decimal("95.000"),
        validation_status="not_available",
        validation_note=None,
        processing_config={"permanent_water": {"source": "JRC Global Surface Water v1.4"}},
        error_code=None,
        error_message_safe=None,
        created_by="pipeline",
        boundary=None,
        summary=AnalysisSummaryOut(),
        available_layers=["study_area", "baseline_flood"],
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class FakeStudyAreaService:
    def __init__(self, areas=None):
        self.areas = areas or []

    def list_areas(self):
        return StudyAreaListResponse(items=[StudyAreaItem(**a) for a in self.areas])

    def get_area(self, slug):
        for area in self.areas:
            if area["slug"] == slug:
                return StudyAreaDetail(**area)
        raise AppError("STUDY_AREA_NOT_FOUND", "Study area was not found.", 404)


class FakeAnalysisService:
    def __init__(self, latest=None, runs=None, metrics=None):
        self.latest_result = latest
        self.runs = runs or []
        self.metrics_result = metrics

    def latest(self, slug):
        if self.latest_result is None:
            raise AppError("ANALYSIS_NOT_FOUND", "No analysis is available for this study area.", 404)
        return self.latest_result

    def list(self, slug, limit=10):
        return SimpleNamespace(items=self.runs, total=len(self.runs))

    def metrics(self, analysis_id):
        if self.metrics_result is None:
            raise AppError("ANALYSIS_NOT_FOUND", "Analysis was not found.", 404)
        return self.metrics_result

    def trigger_analysis(self, request):
        return SimpleNamespace(
            analysis_id=uuid.uuid4(), status="queued", message="Analysis request accepted."
        )


class FakeLayerService:
    def __init__(self, catalogue, geojson=None, available_keys=("baseline_flood",)):
        self.catalogue_result = catalogue
        self.geojson_result = geojson
        self.available_keys = set(available_keys)

    def catalogue(self, analysis_id):
        return self.catalogue_result

    def geojson(self, analysis_id, layer_key):
        if layer_key not in self.available_keys:
            raise AppError("LAYER_NOT_AVAILABLE", "Requested map layer is not available.", 404)
        return self.geojson_result


class FakeReportService:
    def __init__(self, report=None, error=None):
        self.report = report
        self.error = error

    def get_report(self, analysis_id, language):
        if language not in ("bn", "en"):
            raise AppError("UNSUPPORTED_LANGUAGE", "Supported languages are bn and en.", 422)
        if self.error is not None:
            raise self.error
        return self.report


class FakeProvenanceService:
    def __init__(self, sources):
        self.sources_result = sources

    def sources(self, analysis_id):
        return self.sources_result


@pytest.fixture
def latest_response():
    run = make_run()
    return AnalysisLatest(
        id=run.id,
        study_area=StudyAreaRef(slug=run.slug, name=run.name),
        status=run.status,
        analysis_label=run.analysis_label,
        observation_start=run.observation_start,
        observation_end=run.observation_end,
        created_at=run.created_at,
        completed_at=run.completed_at,
        progress_percent=run.progress_percent,
        progress_message=None,
        is_stale=False,
        model=ModelInfo(name=run.model_name, version=run.model_version),
        baseline=BaselineInfo(method=run.baseline_method, display_name="Otsu SAR Thresholding"),
        permanent_water_mask=PermanentWaterMaskInfo(
            applied=True, source="JRC Global Surface Water v1.4", threshold=95.0, note=None
        ),
        validation=ValidationInfo(status="not_available", note=None),
        summary=AnalysisSummaryOut(),
        available_layers=["study_area", "baseline_flood"],
    )


@pytest.fixture
def client(latest_response):
    area = dict(
        id=uuid.uuid4(),
        slug="hakaluki-haor",
        name="Hakaluki Haor",
        division="Sylhet",
        districts=["Moulvibazar"],
        country_code="BD",
        area_km2=None,
        bounds=None,
        is_active=True,
    )
    layer_items = [
        MapLayerItem(
            layer_key="ml_flood",
            display_name="ML-detected flood extent",
            available=False,
            format="geojson",
            area_km2=None,
            feature_count=None,
            source_date="2026-05-13",
            model_version="0.1.0",
            description="Flood-class pixels from custom ML model after configured masking.",
        ),
        MapLayerItem(
            layer_key="baseline_flood",
            display_name="Thresholding baseline",
            available=True,
            format="geojson",
            area_km2=12.5,
            feature_count=3,
            source_date="2026-05-13",
            model_version=None,
            description="Flood-class pixels from documented SAR thresholding baseline.",
        ),
    ]
    app = create_app()
    app.dependency_overrides[get_study_area_service] = lambda: FakeStudyAreaService(areas=[area])
    app.dependency_overrides[get_analysis_runner] = lambda: (lambda **kwargs: None)
    app.dependency_overrides[get_analysis_service] = lambda: FakeAnalysisService(latest=latest_response)
    app.dependency_overrides[get_layer_service] = lambda: FakeLayerService(
        catalogue=LayerCatalogueResponse(analysis_id=latest_response.id, layers=layer_items),
        geojson={
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": "baseline_flood-1",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[91.9, 24.4], [92.0, 24.4], [92.0, 24.5], [91.9, 24.5], [91.9, 24.4]]],
                    },
                    "properties": {"layer_key": "baseline_flood", "area_km2": 12.5, "class_name": "flood"},
                }
            ],
        },
    )
    app.dependency_overrides[get_report_service] = lambda: FakeReportService(
        report=ReportResponse(
            analysis_id=latest_response.id,
            language="bn",
            generation_method="template",
            status="completed",
            title="হাওর এলাকার বন্যা ও কৃষিজমির সম্ভাব্য ঝুঁকি",
            body="সংরক্ষিত বিশ্লেষণ ফলাফল থেকে তৈরি।",
            created_at=datetime(2026, 5, 21, tzinfo=timezone.utc),
            data_as_of=None,
            caveats=["এই ফলাফল সম্ভাব্য কৃষিজমির সংস্পর্শ নির্দেশ করে; নিশ্চিত ফসলের ক্ষতি নয়।"],
        )
    )
    app.dependency_overrides[get_provenance_service] = lambda: FakeProvenanceService(
        sources=SimpleNamespace(
            analysis_id=latest_response.id,
            sources=[
                SimpleNamespace(
                    dataset_key="sentinel1_sar",
                    display_name="Sentinel-1 SAR",
                    provider="Copernicus",
                    role="Flood detection input imagery",
                    version=None,
                    acquired_at=None,
                    period_start=None,
                    period_end=None,
                    resolution_m=10.0,
                    crs=None,
                    source_url=None,
                    status="used",
                    notes=None,
                )
            ],
        )
    )
    return TestClient(app)
