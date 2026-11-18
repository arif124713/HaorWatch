import pytest
from pydantic import ValidationError

from app.schemas.analysis import AnalysisSummaryOut, MethodMetricsOut, ConfusionMatrixOut


def test_area_must_be_non_negative():
    with pytest.raises(ValidationError):
        AnalysisSummaryOut(detected_flood_area_km2=-1.0)


def test_share_percent_bounds():
    with pytest.raises(ValidationError):
        AnalysisSummaryOut(potentially_exposed_agricultural_share_percent=150.0)
    ok = AnalysisSummaryOut(potentially_exposed_agricultural_share_percent=100.0)
    assert ok.potentially_exposed_agricultural_share_percent == 100.0


def test_metric_score_bounds():
    with pytest.raises(ValidationError):
        MethodMetricsOut(method_key="custom_ml", iou=1.5, confusion_matrix=ConfusionMatrixOut())
    with pytest.raises(ValidationError):
        MethodMetricsOut(method_key="custom_ml", precision=-0.1, confusion_matrix=ConfusionMatrixOut())


def test_null_metrics_allowed():
    metrics = MethodMetricsOut(method_key="custom_ml", confusion_matrix=ConfusionMatrixOut())
    assert metrics.iou is None
    summary = AnalysisSummaryOut()
    assert summary.detected_flood_area_km2 is None
