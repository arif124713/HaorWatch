import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]


def execute_run_subprocess(run_id: str, slug: str, start: str, end: str) -> None:
    """Execute a queued analysis run in a subprocess (isolated from the API process)."""
    command = [
        sys.executable,
        "-m",
        "scripts.run_analysis",
        "--slug",
        slug,
        "--start",
        start,
        "--end",
        end,
        "--run-id",
        run_id,
    ]
    try:
        subprocess.run(
            command,
            cwd=str(BACKEND_DIR),
            capture_output=True,
            text=True,
            timeout=7200,
        )
    except Exception:
        # run_pipeline marks the run failed in the database on its own exceptions;
        # a timeout or spawn failure leaves a failed/queued run that the operator
        # can inspect via the API.
        pass
