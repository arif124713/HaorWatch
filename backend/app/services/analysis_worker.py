import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

BACKEND_DIR = Path(__file__).resolve().parents[2]

SUBPROCESS_TIMEOUT_SECONDS = 7200
DB_CONNECT_TIMEOUT_SECONDS = 5

# Patterns that may appear in subprocess output and must never be persisted.
_SECRET_PATTERNS = (
    r"(?i)(api[_-]?key|secret|token|password|passwd|operator[_-]?key|"
    r"x-operator-key|x-goog-api-key|authorization|client[_-]?secret)\s*[:=]\s*\S+",
)


def _redact(text: str | None, limit: int = 2000) -> str:
    """Strip likely secrets from subprocess output before it is stored."""
    if not text:
        return ""
    import re

    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = re.sub(pattern, r"\1=***REDACTED***", redacted)
    return redacted[-limit:]


def _bookkeeping_session_factory():
    """A short-timeout session factory for worker bookkeeping.

    The worker writes run status from the API process. Using a bounded connect
    timeout means an unreachable database fails fast (and is reported) instead of
    hanging the request that launched the background task.
    """
    from app.core.config import get_settings

    engine = create_engine(
        get_settings().database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": DB_CONNECT_TIMEOUT_SECONDS},
    )
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def _mark_run_failed(run_id: str, code: str, message: str, detail: str | None = None) -> None:
    """Record a visible failure on the run so it never stays silently queued."""
    try:
        import uuid as _uuid

        from app.repositories.analysis import AnalysisRepository

        factory = _bookkeeping_session_factory()
        with factory() as db:
            repo = AnalysisRepository(db)
            run = repo.get(_uuid.UUID(run_id))
            if run is None or run.status in ("completed", "failed"):
                return
            note = _redact(detail)
            repo.update_run(
                run,
                status="failed",
                error_code=code,
                error_message_safe=message,
                completed_at=datetime.now(timezone.utc),
                progress_message=None,
                processing_config={
                    **(run.processing_config or {}),
                    "subprocess_failure": {"code": code, "message": message, "detail": note},
                },
            )
    except Exception:  # noqa: BLE001 - failure reporting must never raise
        pass


def execute_run_subprocess(
    run_id: str,
    slug: str,
    start: str | None,
    end: str | None,
    prepared_dir: str | None = None,
) -> None:
    """Execute a queued analysis run in a subprocess (isolated from the API process).

    Records the subprocess exit code, stdout, stderr, and timeout outcome so that
    failures are visible through the API. If the subprocess fails to launch,
    times out, or exits non-zero, the run is marked ``failed`` with a safe code.
    A run that is already ``processing``/``completed``/``failed`` is not
    re-executed, preventing duplicate work.
    """
    from app.repositories.analysis import AnalysisRepository
    import uuid as _uuid

    # Duplicate-execution guard: only a queued run may start.
    try:
        factory = _bookkeeping_session_factory()
        with factory() as db:
            repo = AnalysisRepository(db)
            run = repo.get(_uuid.UUID(run_id))
            if run is None:
                return
            if run.status != "queued":
                return
            repo.update_run(run, status="processing", progress_percent=1,
                            progress_message="Subprocess launched")
    except Exception:  # noqa: BLE001
        pass

    command = [sys.executable, "-m", "scripts.run_analysis", "--slug", slug, "--run-id", run_id]
    if prepared_dir:
        command += ["--prepared-dir", prepared_dir]
    else:
        command += ["--start", start, "--end", end]

    try:
        completed = subprocess.run(
            command,
            cwd=str(BACKEND_DIR),
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        _mark_run_failed(
            run_id,
            "SUBPROCESS_TIMEOUT",
            f"Analysis subprocess exceeded {SUBPROCESS_TIMEOUT_SECONDS}s.",
            detail=(exc.stderr or "") if isinstance(exc.stderr, str) else None,
        )
        return
    except OSError as exc:
        _mark_run_failed(run_id, "SUBPROCESS_LAUNCH_FAILED", "Analysis subprocess could not be launched.",
                         detail=str(exc))
        return

    if completed.returncode != 0:
        detail = _redact(completed.stderr)
        _mark_run_failed(
            run_id,
            "SUBPROCESS_NONZERO_EXIT",
            f"Analysis subprocess exited with code {completed.returncode}.",
            detail=detail,
        )
