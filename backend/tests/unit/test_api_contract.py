import uuid

from tests.unit.conftest import make_run

from app.schemas.analysis import (
    AnalysisDetail,
    AnalysisSummaryOut,
    BaselineInfo,
    ConfusionMatrixOut,
    MetricsResponse,
    MethodMetricsOut,
    ModelInfo,
    PermanentWaterMaskInfo,
    StudyAreaRef,
    TestDatasetOut,
    ValidationInfo,
)


def test_latest_analysis_contract(client, latest_response):
    response = client.get("/api/v1/study-areas/hakaluki-haor/analyses/latest")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(latest_response.id)
    assert body["status"] == "completed"
    assert body["is_stale"] is False
    assert set(body["summary"].keys()) == {
        "study_area_km2",
        "detected_flood_area_km2",
        "baseline_flood_area_km2",
        "mapped_agricultural_area_km2",
        "potentially_exposed_agricultural_area_km2",
        "potentially_exposed_agricultural_share_percent",
        "valid_pixel_count",
        "nodata_pixel_count",
    }
    assert body["model"]["name"] == "HaorWatch Flood Classifier"
    assert body["baseline"]["method"] == "otsu_thresholding"
    assert body["permanent_water_mask"]["applied"] is True
    assert body["available_layers"] == ["study_area", "baseline_flood"]


def test_study_areas_contract(client):
    response = client.get("/api/v1/study-areas")
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 1
    item = body["items"][0]
    assert item["slug"] == "hakaluki-haor"
    assert item["country_code"] == "BD"


def test_analysis_not_found_envelope(client):
    app = client.app
    from app.api.deps import get_analysis_service
    from tests.unit.conftest import FakeAnalysisService

    app.dependency_overrides[get_analysis_service] = lambda: FakeAnalysisService(latest=None)
    response = client.get("/api/v1/study-areas/hakaluki-haor/analyses/latest")
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "ANALYSIS_NOT_FOUND"
    assert "message" in body["error"]


def test_layer_catalogue_contract(client):
    response = client.get(f"/api/v1/analyses/{uuid.uuid4()}/layers")
    assert response.status_code == 200
    body = response.json()
    assert body["layers"][0]["layer_key"] == "ml_flood"
    assert body["layers"][0]["available"] is False
    assert body["layers"][0]["format"] == "geojson"


def test_layer_geojson_media_type(client):
    response = client.get(f"/api/v1/analyses/{uuid.uuid4()}/layers/baseline_flood")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/geo+json")
    body = response.json()
    assert body["type"] == "FeatureCollection"
    assert body["features"][0]["properties"]["layer_key"] == "baseline_flood"


def test_layer_not_available(client):
    response = client.get(f"/api/v1/analyses/{uuid.uuid4()}/layers/ml_flood")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "LAYER_NOT_AVAILABLE"


def test_metrics_contract(client, latest_response):
    app = client.app
    from app.api.deps import get_analysis_service
    from tests.unit.conftest import FakeAnalysisService

    metrics = MetricsResponse(
        analysis_id=latest_response.id,
        validation_status="internal_test_only",
        test_dataset=TestDatasetOut(
            name="Sen1Floods11", version="v1.1", split_strategy="sen1floods11_official_event_split", test_pixel_count=1000
        ),
        methods=[
            MethodMetricsOut(
                method_key="custom_ml",
                method_name="Custom ML Model",
                precision=None,
                recall=None,
                f1=None,
                iou=None,
                confusion_matrix=ConfusionMatrixOut(),
            ),
            MethodMetricsOut(
                method_key="otsu_baseline",
                method_name="Otsu SAR Thresholding",
                precision=None,
                recall=None,
                f1=None,
                iou=None,
                confusion_matrix=ConfusionMatrixOut(),
            ),
        ],
    )
    app.dependency_overrides[get_analysis_service] = lambda: FakeAnalysisService(latest=latest_response, metrics=metrics)
    response = client.get(f"/api/v1/analyses/{uuid.uuid4()}/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["positive_class"] == "flood"
    assert body["validation_status"] == "internal_test_only"
    assert [m["method_key"] for m in body["methods"]] == ["custom_ml", "otsu_baseline"]
    assert body["methods"][0]["precision"] is None


def test_report_bn_contract(client):
    response = client.get("/api/v1/analyses/{}/reports/bn".format(uuid.uuid4()))
    assert response.status_code == 200
    body = response.json()
    assert body["language"] == "bn"
    assert body["generation_method"] in ("template", "llm")
    assert len(body["caveats"]) > 0


def test_unsupported_language(client):
    response = client.get("/api/v1/analyses/{}/reports/fr".format(uuid.uuid4()))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "UNSUPPORTED_LANGUAGE"


def test_admin_requires_operator_key(client):
    response = client.post("/api/v1/admin/analyses", json={"study_area_slug": "hakaluki-haor"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "INVALID_OPERATOR_KEY"


def test_admin_wrong_key_rejected(client):
    response = client.post(
        "/api/v1/admin/analyses",
        json={"study_area_slug": "hakaluki-haor"},
        headers={"X-Operator-Key": "wrong"},
    )
    assert response.status_code == 401


def test_admin_accepts_request(client, monkeypatch):
    monkeypatch.setenv("OPERATOR_API_KEY", "test-secret-123")
    from app.core.config import get_settings

    get_settings.cache_clear()
    app = client.app
    from app.api.deps import get_settings_dep

    app.dependency_overrides[get_settings_dep] = lambda: get_settings()
    try:
        response = client.post(
            "/api/v1/admin/analyses",
            json={"study_area_slug": "hakaluki-haor", "observation_start": "2026-05-01T00:00:00Z",
                  "observation_end": "2026-05-15T00:00:00Z"},
            headers={"X-Operator-Key": "test-secret-123"},
        )
        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "queued"
        assert body["message"] == "Analysis request accepted."
    finally:
        get_settings.cache_clear()


def test_health_contract(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "haorwatch-api"
    assert "timestamp" in body
