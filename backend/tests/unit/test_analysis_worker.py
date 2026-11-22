import subprocess
import uuid
from types import SimpleNamespace

import pytest

from app.services import analysis_worker

RUN_ID = str(uuid.uuid4())


class _Ctx:
    def __enter__(self):
        return object()

    def __exit__(self, *exc):
        return False


def _install_fake_db(monkeypatch, run):
    """Patch the session factory and repository used by the worker; return updates list."""
    updates = []

    class _FakeAnalysisRepository:
        def __init__(self, _db):
            pass

        def get(self, _run_id):
            return run

        def update_run(self, run, **fields):
            updates.append(fields)
            for key, value in fields.items():
                setattr(run, key, value)
            return run

    monkeypatch.setattr("app.repositories.analysis.AnalysisRepository", _FakeAnalysisRepository)
    monkeypatch.setattr(analysis_worker, "_bookkeeping_session_factory", lambda: (lambda: _Ctx()))
    return updates


def _codes(updates):
    return [u.get("error_code") for u in updates if u.get("error_code")]


def test_nonzero_exit_marks_run_failed(monkeypatch):
    run = SimpleNamespace(id=RUN_ID, status="queued", processing_config={})
    updates = _install_fake_db(monkeypatch, run)
    monkeypatch.setattr(
        analysis_worker.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(args=[], returncode=2, stdout="", stderr="boom"),
    )
    analysis_worker.execute_run_subprocess(RUN_ID, "hakaluki-haor", "2026-05-01", "2026-05-02")
    assert "SUBPROCESS_NONZERO_EXIT" in _codes(updates)
    assert run.status == "failed"


def test_timeout_marks_run_failed(monkeypatch):
    run = SimpleNamespace(id=RUN_ID, status="queued", processing_config={})
    updates = _install_fake_db(monkeypatch, run)

    def fake_run(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="x", timeout=7200)

    monkeypatch.setattr(analysis_worker.subprocess, "run", fake_run)
    analysis_worker.execute_run_subprocess(RUN_ID, "hakaluki-haor", "2026-05-01", "2026-05-02")
    assert "SUBPROCESS_TIMEOUT" in _codes(updates)
    assert run.status == "failed"


def test_launch_failure_marks_run_failed(monkeypatch):
    run = SimpleNamespace(id=RUN_ID, status="queued", processing_config={})
    updates = _install_fake_db(monkeypatch, run)

    def fake_run(*_a, **_k):
        raise OSError("cannot spawn")

    monkeypatch.setattr(analysis_worker.subprocess, "run", fake_run)
    analysis_worker.execute_run_subprocess(RUN_ID, "hakaluki-haor", "2026-05-01", "2026-05-02")
    assert "SUBPROCESS_LAUNCH_FAILED" in _codes(updates)
    assert run.status == "failed"


def test_non_queued_run_is_not_re_executed(monkeypatch):
    run = SimpleNamespace(id=RUN_ID, status="processing", processing_config={})
    _install_fake_db(monkeypatch, run)
    called = {"ran": False}

    def fake_run(*_a, **_k):
        called["ran"] = True
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(analysis_worker.subprocess, "run", fake_run)
    analysis_worker.execute_run_subprocess(RUN_ID, "hakaluki-haor", "2026-05-01", "2026-05-02")
    assert called["ran"] is False


def test_successful_run_is_marked_processing_not_failed(monkeypatch):
    run = SimpleNamespace(id=RUN_ID, status="queued", processing_config={})
    updates = _install_fake_db(monkeypatch, run)
    monkeypatch.setattr(
        analysis_worker.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr=""),
    )
    analysis_worker.execute_run_subprocess(RUN_ID, "hakaluki-haor", "2026-05-01", "2026-05-02")
    assert _codes(updates) == []
    assert run.status == "processing"


def test_redact_strips_likely_secrets():
    text = "OPERATOR_API_KEY=hwop_deadbeef\nx-goog-api-key: AIza12345\nnormal log line"
    redacted = analysis_worker._redact(text)
    assert "hwop_deadbeef" not in redacted
    assert "AIza12345" not in redacted
    assert "REDACTED" in redacted
    assert "normal log line" in redacted


def test_nonzero_exit_detail_is_redacted(monkeypatch):
    run = SimpleNamespace(id=RUN_ID, status="queued", processing_config={})
    updates = _install_fake_db(monkeypatch, run)
    monkeypatch.setattr(
        analysis_worker.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="secret=abcdef\ntraceback follows"
        ),
    )
    analysis_worker.execute_run_subprocess(RUN_ID, "hakaluki-haor", "2026-05-01", "2026-05-02")
    failure = [u for u in updates if u.get("error_code") == "SUBPROCESS_NONZERO_EXIT"][0]
    detail = failure["processing_config"]["subprocess_failure"]["detail"]
    assert "abcdef" not in detail
    assert "REDACTED" in detail
