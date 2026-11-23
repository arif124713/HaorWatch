import logging
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger("haorwatch")

BACKEND_DIR = Path(__file__).resolve().parents[2]

SUBPROCESS_TIMEOUT_SECONDS = 7200
DB_CONNECT_TIMEOUT_SECONDS = 5
# Bounded excerpt length for any subprocess output persisted to the database.
MAX_LOG_EXCERPT = 2000
REDACTED = "***REDACTED***"

# Redaction strategy:
# 1. Credential-bearing headers (Authorization: Bearer/Basic X, X-Operator-Key,
#    x-goog-api-key, api-key, etc.) -> replace the *value* after the separator.
# 2. key=value / key: value where the key name looks like a secret -> replace
#    only the value, preserving the key for diagnostics.
# 3. URL query parameters whose name looks like a credential -> replace value.
_HEADER_PATTERNS = (
    # Authorization: Bearer <token> / Authorization: Basic <b64>
    re.compile(r"(?i)(authorization\s*:\s*)(bearer|basic|token)\s+[^\s,;\"']+"),
    # Authorization: <token> (no scheme, and not already redacted/schemed)
    re.compile(r"(?i)(authorization\s*[:=]\s*)(?!(?:bearer|basic|token|\*{3}REDACTED))[^\s,;\"']+"),
    # Standalone bearer/basic tokens not preceded by a header name
    re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._\-+/=]{8,}"),
    # Named credential headers, e.g. X-Operator-Key: abc, x-goog-api-key=abc
    re.compile(
        r"(?i)((?:x-)?(?:operator|goog|api|x-api)[-_]?key\s*[:=]\s*)[^\s,;\"']+"
    ),
)
_KV_PATTERN = re.compile(
    r"""(?ix)
    (                                 # key group (quoted or bare)
      ["']?[A-Za-z0-9_.\-]*            #   any name containing...
      (?:secret|token|passwo?rd|passwd|api[_-]?key|access[_-]?key|
         private[_-]?key|operator[_-]?key|client[_-]?secret|
         credential|bearer)
      [A-Za-z0-9_.\-]*["']?
    )
    (\s*[:=]\s*)                       # separator
    (                                  # value
      "(?:[^"\\]|\\.)*" | '(?:[^'\\]|\\.)*' | [^\s,;&]+
    )
    """
)
_URL_QUERY_PATTERN = re.compile(
    r"(?i)([?&](?:token|key|api[_-]?key|secret|access[_-]?token|"
    r"client[_-]?secret|password|passwd|auth)=)[^&#\s]+"
)


def _to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _redact(value, limit: int = MAX_LOG_EXCERPT) -> str:
    """Sanitize subprocess text and bound its length.

    Removes credential *values* (never just the scheme/header name) for headers,
    key/value pairs, and URL query parameters, then truncates to ``limit``.
    ``bytes`` input (e.g. from ``TimeoutExpired``) is decoded first.
    """
    text = _to_text(value)
    if not text:
        return ""
    # Bound the input BEFORE running regexes: subprocess output can be huge and
    # unbounded regexes over a large blob are slow (and a DoS vector). Redaction
    # then operates on at most ~2x the persisted limit.
    if len(text) > limit * 2:
        text = text[-limit * 2:]
    # 1. URL query parameters carrying credentials (before generic key/value so
    #    the value class does not swallow subsequent &-separated params).
    text = _URL_QUERY_PATTERN.sub(lambda m: f"{m.group(1)}{REDACTED}", text)
    # 2. Authorization: Bearer/Basic <token>, Authorization: <token>
    text = _HEADER_PATTERNS[0].sub(lambda m: f"{m.group(1)}{m.group(2)} {REDACTED}", text)
    text = _HEADER_PATTERNS[1].sub(lambda m: f"{m.group(1)}{REDACTED}", text)
    # 3. Standalone bearer/basic tokens.
    text = _HEADER_PATTERNS[2].sub(lambda m: f"{m.group(1)} {REDACTED}", text)
    # 4. Named credential headers (X-Operator-Key, x-goog-api-key, ...).
    text = _HEADER_PATTERNS[3].sub(lambda m: f"{m.group(1)}{REDACTED}", text)
    # 5. key=value / key: value where the key name looks like a secret.
    text = _KV_PATTERN.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)
    return text[-limit:]


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


def _outcome_block(block: dict) -> dict:
    """Merge a subprocess outcome block into a run's processing_config."""
    return {"subprocess": block}


def _record_outcome(run_id: str, block: dict, mark_failed: bool, code: str | None = None,
                    message: str | None = None) -> None:
    """Persist the sanitized subprocess outcome on the run.

    When ``mark_failed`` is set, the run is marked ``failed`` unless it is already
    ``completed``/``failed`` (a late worker update must never overwrite a run the
    pipeline already finalised). Database bookkeeping failures are logged but not
    raised, so the request that launched the worker is not broken.
    """
    try:
        import uuid as _uuid

        from app.repositories.analysis import AnalysisRepository

        factory = _bookkeeping_session_factory()
        with factory() as db:
            repo = AnalysisRepository(db)
            run = repo.get(_uuid.UUID(run_id))
            if run is None:
                logger.warning("subprocess outcome for unknown run %s", run_id)
                return
            if run.status in ("completed", "failed"):
                # Do not overwrite a finalised run; leave a diagnostic trace only.
                logger.info("ignoring subprocess outcome for finalised run %s (%s)", run_id, run.status)
                return
            fields = {"processing_config": {**(run.processing_config or {}), **_outcome_block(block)}}
            if mark_failed:
                fields.update(
                    status="failed",
                    error_code=code,
                    error_message_safe=message,
                    completed_at=datetime.now(timezone.utc),
                    progress_message=None,
                )
            repo.update_run(run, **fields)
    except Exception:  # noqa: BLE001 - bookkeeping must never raise into the request
        logger.exception("failed to record subprocess outcome for run %s", run_id)


def execute_run_subprocess(
    run_id: str,
    slug: str,
    start: str | None,
    end: str | None,
    prepared_dir: str | None = None,
) -> None:
    """Execute a queued analysis run in a subprocess (isolated from the API process).

    Behaviour contract:
    * only a ``queued`` run is executed; ``processing``/``completed``/``failed``
      runs are never re-executed (duplicate-execution guard);
    * the subprocess exit code, timeout/launch outcome, sanitized and bounded
      stdout/stderr excerpts, failure code, and finish timestamp are recorded on
      the run under ``processing_config.subprocess``;
    * a non-zero exit, timeout, or launch failure marks the run ``failed`` with a
      safe error code, unless the run was already finalised by the pipeline;
    * credentials are redacted before any output is persisted.
    """
    from app.repositories.analysis import AnalysisRepository
    import uuid as _uuid

    finished_at = lambda: datetime.now(timezone.utc).isoformat()  # noqa: E731

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
        logger.exception("subprocess guard bookkeeping failed for run %s", run_id)

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
        _record_outcome(
            run_id,
            {
                "status": "timeout",
                "timed_out": True,
                "exit_code": None,
                "launched": True,
                "failure_code": "SUBPROCESS_TIMEOUT",
                "finished_at": finished_at(),
                "stdout_excerpt": _redact(exc.stdout),
                "stderr_excerpt": _redact(exc.stderr),
            },
            mark_failed=True,
            code="SUBPROCESS_TIMEOUT",
            message=f"Analysis subprocess exceeded {SUBPROCESS_TIMEOUT_SECONDS}s.",
        )
        return
    except OSError as exc:
        _record_outcome(
            run_id,
            {
                "status": "launch_failed",
                "timed_out": False,
                "exit_code": None,
                "launched": False,
                "failure_code": "SUBPROCESS_LAUNCH_FAILED",
                "finished_at": finished_at(),
                "stdout_excerpt": "",
                "stderr_excerpt": _redact(str(exc)),
            },
            mark_failed=True,
            code="SUBPROCESS_LAUNCH_FAILED",
            message="Analysis subprocess could not be launched.",
        )
        return

    if completed.returncode == 0:
        _record_outcome(
            run_id,
            {
                "status": "completed",
                "timed_out": False,
                "exit_code": 0,
                "launched": True,
                "failure_code": None,
                "finished_at": finished_at(),
                "stdout_excerpt": _redact(completed.stdout),
                "stderr_excerpt": _redact(completed.stderr),
            },
            mark_failed=False,
        )
        return

    _record_outcome(
        run_id,
        {
            "status": "failed",
            "timed_out": False,
            "exit_code": completed.returncode,
            "launched": True,
            "failure_code": "SUBPROCESS_NONZERO_EXIT",
            "finished_at": finished_at(),
            "stdout_excerpt": _redact(completed.stdout),
            "stderr_excerpt": _redact(completed.stderr),
        },
        mark_failed=True,
        code="SUBPROCESS_NONZERO_EXIT",
        message=f"Analysis subprocess exited with code {completed.returncode}.",
    )
