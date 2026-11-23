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


def _subprocess_block(updates):
    for u in updates:
        block = u.get("processing_config", {}).get("subprocess")
        if block:
            return block
    return None


def _codes(updates):
    return [u.get("error_code") for u in updates if u.get("error_code")]


def test_nonzero_exit_marks_run_failed(monkeypatch):
    run = SimpleNamespace(id=RUN_ID, status="queued", processing_config={})
    updates = _install_fake_db(monkeypatch, run)
    monkeypatch.setattr(
        analysis_worker.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(args=[], returncode=2, stdout="out", stderr="boom"),
    )
    analysis_worker.execute_run_subprocess(RUN_ID, "hakaluki-haor", "2026-05-01", "2026-05-02")
    assert "SUBPROCESS_NONZERO_EXIT" in _codes(updates)
    assert run.status == "failed"
    block = _subprocess_block(updates)
    assert block["exit_code"] == 2
    assert block["status"] == "failed"
    assert block["finished_at"]


def test_timeout_marks_run_failed_and_sanitizes(monkeypatch):
    run = SimpleNamespace(id=RUN_ID, status="queued", processing_config={})
    updates = _install_fake_db(monkeypatch, run)

    def fake_run(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="x", timeout=7200, output=b"Authorization: Bearer TIMEOUTTOKEN")

    monkeypatch.setattr(analysis_worker.subprocess, "run", fake_run)
    analysis_worker.execute_run_subprocess(RUN_ID, "hakaluki-haor", "2026-05-01", "2026-05-02")
    assert "SUBPROCESS_TIMEOUT" in _codes(updates)
    assert run.status == "failed"
    block = _subprocess_block(updates)
    assert block["timed_out"] is True
    assert "TIMEOUTTOKEN" not in block["stdout_excerpt"]


def test_timeout_bytes_stderr_is_handled(monkeypatch):
    run = SimpleNamespace(id=RUN_ID, status="queued", processing_config={})
    updates = _install_fake_db(monkeypatch, run)

    def fake_run(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="x", timeout=7200, output=b"line1", stderr=b"secret=BYTESECRET")

    monkeypatch.setattr(analysis_worker.subprocess, "run", fake_run)
    analysis_worker.execute_run_subprocess(RUN_ID, "hakaluki-haor", "2026-05-01", "2026-05-02")
    block = _subprocess_block(updates)
    assert "BYTESECRET" not in block["stderr_excerpt"]


def test_launch_failure_marks_run_failed(monkeypatch):
    run = SimpleNamespace(id=RUN_ID, status="queued", processing_config={})
    updates = _install_fake_db(monkeypatch, run)

    def fake_run(*_a, **_k):
        raise OSError("cannot spawn")

    monkeypatch.setattr(analysis_worker.subprocess, "run", fake_run)
    analysis_worker.execute_run_subprocess(RUN_ID, "hakaluki-haor", "2026-05-01", "2026-05-02")
    assert "SUBPROCESS_LAUNCH_FAILED" in _codes(updates)
    assert run.status == "failed"
    block = _subprocess_block(updates)
    assert block["launched"] is False


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


def test_completed_run_is_not_overwritten_by_late_worker(monkeypatch):
    # The pipeline already finalised the run; a late worker outcome must not fail it.
    run = SimpleNamespace(id=RUN_ID, status="completed", processing_config={})
    updates = _install_fake_db(monkeypatch, run)
    monkeypatch.setattr(
        analysis_worker.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="boom"),
    )
    analysis_worker.execute_run_subprocess(RUN_ID, "hakaluki-haor", "2026-05-01", "2026-05-02")
    assert run.status == "completed"
    assert _codes(updates) == []


def test_successful_run_records_diagnostics_without_failing(monkeypatch):
    run = SimpleNamespace(id=RUN_ID, status="queued", processing_config={})
    updates = _install_fake_db(monkeypatch, run)
    monkeypatch.setattr(
        analysis_worker.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(args=[], returncode=0, stdout="ok done", stderr=""),
    )
    analysis_worker.execute_run_subprocess(RUN_ID, "hakaluki-haor", "2026-05-01", "2026-05-02")
    assert _codes(updates) == []
    block = _subprocess_block(updates)
    assert block["status"] == "completed"
    assert block["exit_code"] == 0
    assert block["stdout_excerpt"] == "ok done"
    assert run.status == "processing"  # status unchanged by a successful worker bookkeeping


def test_missing_run_does_not_launch(monkeypatch):
    class _Repo:
        def __init__(self, _db):
            pass

        def get(self, _run_id):
            return None

    monkeypatch.setattr("app.repositories.analysis.AnalysisRepository", _Repo)
    monkeypatch.setattr(analysis_worker, "_bookkeeping_session_factory", lambda: (lambda: _Ctx()))
    called = {"ran": False}
    monkeypatch.setattr(analysis_worker.subprocess, "run",
                        lambda *a, **k: called.__setitem__("ran", True))
    analysis_worker.execute_run_subprocess(RUN_ID, "hakaluki-haor", "2026-05-01", "2026-05-02")
    assert called["ran"] is False


def test_db_bookkeeping_failure_does_not_raise(monkeypatch):
    monkeypatch.setattr(analysis_worker, "_bookkeeping_session_factory",
                        lambda: (_ for _ in ()).throw(RuntimeError("db down")))
    monkeypatch.setattr(
        analysis_worker.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
    )
    # Must not raise even though bookkeeping fails.
    analysis_worker.execute_run_subprocess(RUN_ID, "hakaluki-haor", "2026-05-01", "2026-05-02")


def test_log_excerpt_is_bounded(monkeypatch):
    run = SimpleNamespace(id=RUN_ID, status="queued", processing_config={})
    updates = _install_fake_db(monkeypatch, run)
    monkeypatch.setattr(
        analysis_worker.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(args=[], returncode=0, stdout="y" * 50000, stderr=""),
    )
    analysis_worker.execute_run_subprocess(RUN_ID, "hakaluki-haor", "2026-05-01", "2026-05-02")
    block = _subprocess_block(updates)
    assert len(block["stdout_excerpt"]) <= analysis_worker.MAX_LOG_EXCERPT


# --- redaction coverage -----------------------------------------------------

@pytest.mark.parametrize(
    "line,secret",
    [
        ("API_KEY=abc123secret", "abc123secret"),
        ("SECRET=s3cr3t", "s3cr3t"),
        ("PASSWORD=hunter2", "hunter2"),
        ("OPERATOR_API_KEY=hwop_deadbeef", "hwop_deadbeef"),
        ("X-Operator-Key: hwop_1234567890", "hwop_1234567890"),
        ("x-goog-api-key: AIzaSyABCDEF", "AIzaSyABCDEF"),
        ("Authorization: Bearer ACTUAL_SECRET_TOKEN", "ACTUAL_SECRET_TOKEN"),
        ("Authorization: Basic dXNlcjpwYXNz", "dXNlcjpwYXNz"),
        ("client_secret=zzz999", "zzz999"),
        ("url https://x.example/y?token=abc123&other=keep", "abc123"),
        ('{"api_key": "jsonsecret"}', "jsonsecret"),
    ],
)
def test_redact_removes_secret_values(line, secret):
    redacted = analysis_worker._redact(line)
    assert secret not in redacted
    assert analysis_worker.REDACTED in redacted


@pytest.mark.parametrize(
    "line",
    [
        "normal log line",
        "value=5 status=ok",
        "duration=1.5s",
        "https://example.com/plain?page=2",
    ],
)
def test_redact_preserves_non_secret_diagnostics(line):
    assert analysis_worker._redact(line) == line


def test_redact_handles_bytes_and_none():
    assert analysis_worker._redact(None) == ""
    assert "TOKEN" not in analysis_worker._redact(b"Authorization: Bearer TOKEN")
