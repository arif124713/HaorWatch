from datetime import datetime, timezone
import uuid

from app.services.report_service import ReportService


class FakeRepo:
    def __init__(self, run, summary=None):
        self.run = run
        self.summary = summary

    def get(self, analysis_id):
        return self.run

    def summary_of(self, run_id):
        return self.summary

    def report_of(self, run_id, language):
        return None

    def save_report(self, report):
        return report


SUMMARY_FIELDS = [
    "study_area_km2",
    "detected_flood_area_km2",
    "baseline_flood_area_km2",
    "mapped_agricultural_area_km2",
    "potentially_exposed_agricultural_area_km2",
    "potentially_exposed_agricultural_share_percent",
    "valid_pixel_count",
    "nodata_pixel_count",
]


def make_run(summary_values=None):
    from types import SimpleNamespace

    fields = {name: None for name in SUMMARY_FIELDS}
    fields.update(summary_values or {})
    summary = SimpleNamespace(**fields)
    return SimpleNamespace(
        id=uuid.uuid4(),
        study_area=SimpleNamespace(name="Hakaluki Haor", slug="hakaluki-haor"),
        status="completed",
        observation_start=datetime(2026, 5, 1, tzinfo=timezone.utc),
        observation_end=datetime(2026, 5, 13, tzinfo=timezone.utc),
        completed_at=datetime(2026, 5, 21, tzinfo=timezone.utc),
        model_name="HaorWatch Flood Classifier",
        model_version="0.1.0",
        baseline_method="otsu_thresholding",
        permanent_water_mask_applied=True,
        permanent_water_threshold=None,
        validation_status="not_available",
        validation_note=None,
    ), summary


def test_template_bn_uses_stored_values_and_marks_missing():
    run, summary = make_run(
        dict(
            study_area_km2=250.0,
            detected_flood_area_km2=42.567,
            baseline_flood_area_km2=None,
            mapped_agricultural_area_km2=80.0,
            potentially_exposed_agricultural_area_km2=12.34,
            potentially_exposed_agricultural_share_percent=15.425,
        )
    )
    service = ReportService(repo=FakeRepo(run, summary), llm=None)
    report = service.get_report(run.id, "bn")
    assert "42.57 বর্গ কিলোমিটার" in report.body
    assert "12.34 বর্গ কিলোমিটার" in report.body
    assert "15.43%" in report.body
    assert "পাওয়া যায়নি" in report.body
    assert report.generation_method == "template"
    assert report.title == "হাওর এলাকার বন্যা ও কৃষিজমির সম্ভাব্য ঝুঁকি"


def test_template_en_marks_missing_values():
    run, summary = make_run(dict())
    service = ReportService(repo=FakeRepo(run, summary), llm=None)
    report = service.get_report(run.id, "en")
    assert "Not available" in report.body
    assert "Hakaluki Haor" in report.body


def test_all_null_summary_has_no_invented_numbers():
    run, summary = make_run()
    service = ReportService(repo=FakeRepo(run, summary), llm=None)
    report = service.get_report(run.id, "bn")
    digits = [token for token in report.body.split() if token.replace(",", "").replace(".", "").isdigit()]
    assert digits == []


def test_caveats_present_and_validation_note():
    run, summary = make_run()
    service = ReportService(repo=FakeRepo(run, summary), llm=None)
    report = service.get_report(run.id, "bn")
    assert any("সম্ভাব্য" in caveat for caveat in report.caveats)
    assert any("যাচাই" in caveat for caveat in report.caveats)
