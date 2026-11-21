import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.core.exceptions import AppError
from app.schemas.analysis import TriggerAnalysisRequest
from app.services.analysis_service import AnalysisService

RUN_ID = uuid.uuid4()


class _Scalars:
    def __init__(self, value):
        self._value = value

    def first(self):
        return self._value


class _FakeDb:
    def __init__(self, busy=None):
        self._busy = busy
        self.created = []

    def scalars(self, _stmt):
        return _Scalars(self._busy)


class _FakeRepo:
    def __init__(self, busy=None):
        self.db = _FakeDb(busy=busy)

    def create_run(self, **fields):
        run = SimpleNamespace(id=RUN_ID, **fields)
        self.db.created.append(fields)
        return run


class _FakeAreaRepo:
    def __init__(self, area):
        self._area = area

    def get_by_slug(self, slug):
        return self._area if self._area and self._area.slug == slug else None


def _service(busy=None, area=SimpleNamespace(slug="hakaluki-haor", id="area-1")):
    return AnalysisService(db=None, repo=_FakeRepo(busy=busy), area_repo=_FakeAreaRepo(area))


def test_missing_dates_rejected_for_live_run():
    service = _service()
    with pytest.raises(AppError) as exc:
        service.trigger_analysis(TriggerAnalysisRequest(study_area_slug="hakaluki-haor"))
    assert exc.value.code == "OBSERVATION_WINDOW_REQUIRED"
    assert service.repo.db.created == []


def test_only_start_date_rejected():
    service = _service()
    with pytest.raises(AppError) as exc:
        service.trigger_analysis(TriggerAnalysisRequest(
            study_area_slug="hakaluki-haor",
            observation_start=datetime(2026, 5, 1, tzinfo=timezone.utc),
        ))
    assert exc.value.code == "OBSERVATION_WINDOW_REQUIRED"


def test_invalid_window_rejected():
    service = _service()
    with pytest.raises(AppError) as exc:
        service.trigger_analysis(TriggerAnalysisRequest(
            study_area_slug="hakaluki-haor",
            observation_start=datetime(2026, 5, 10, tzinfo=timezone.utc),
            observation_end=datetime(2026, 5, 1, tzinfo=timezone.utc),
        ))
    assert exc.value.code == "INVALID_OBSERVATION_WINDOW"


def test_valid_dates_accept_and_queue():
    service = _service()
    accepted = service.trigger_analysis(TriggerAnalysisRequest(
        study_area_slug="hakaluki-haor",
        observation_start=datetime(2026, 5, 1, tzinfo=timezone.utc),
        observation_end=datetime(2026, 5, 15, tzinfo=timezone.utc),
    ))
    assert accepted.status == "queued"
    assert service.repo.db.created[0]["status"] == "queued"


def test_duplicate_trigger_rejected_when_busy():
    service = _service(busy=SimpleNamespace(status="processing", id="run-busy"))
    with pytest.raises(AppError) as exc:
        service.trigger_analysis(TriggerAnalysisRequest(
            study_area_slug="hakaluki-haor",
            observation_start=datetime(2026, 5, 1, tzinfo=timezone.utc),
            observation_end=datetime(2026, 5, 15, tzinfo=timezone.utc),
        ))
    assert exc.value.code == "ANALYSIS_ALREADY_RUNNING"


def test_prepared_dir_requires_valid_bundle(tmp_path):
    service = _service()
    with pytest.raises(AppError) as exc:
        service.trigger_analysis(TriggerAnalysisRequest(
            study_area_slug="hakaluki-haor", prepared_dir=str(tmp_path / "missing"),
        ))
    assert exc.value.code == "PREPARED_BUNDLE_INVALID"


def test_prepared_dir_accepts_existing_bundle(tmp_path):
    (tmp_path / "s1.tif").write_bytes(b"stub")
    service = _service()
    accepted = service.trigger_analysis(TriggerAnalysisRequest(
        study_area_slug="hakaluki-haor", prepared_dir=str(tmp_path),
    ))
    assert accepted.status == "queued"
    assert service.repo.db.created[0]["processing_config"]["execution_mode"] == "prepared"