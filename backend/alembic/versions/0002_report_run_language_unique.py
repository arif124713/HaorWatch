"""reports: unique (analysis_run_id, language)

Adds a database-level unique constraint so concurrent report generation cannot
persist duplicate reports for the same analysis run and language.

Revision ID: 0002_report_run_language_unique
Revises: 0001_initial
Create Date: 2026-11-22
"""

from alembic import op

revision = "0002_report_run_language_unique"
down_revision = "0001_initial"
branch_labels = None
depends_on = None

CONSTRAINT_NAME = "uq_report_run_language"


def upgrade() -> None:
    # De-duplicate before adding the constraint. Existing rows are kept
    # deterministically: for each (analysis_run_id, language) we keep the
    # earliest report (by created_at, then id) and remove the later duplicates,
    # which are by definition redundant copies of the same report. No non-report
    # data is touched.
    op.execute(
        """
        DELETE FROM reports
        WHERE id IN (
            SELECT id FROM (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY analysis_run_id, language
                           ORDER BY created_at ASC NULLS LAST, id ASC
                       ) AS rn
                FROM reports
            ) ranked
            WHERE ranked.rn > 1
        )
        """
    )
    op.create_unique_constraint(
        CONSTRAINT_NAME, "reports", ["analysis_run_id", "language"]
    )


def downgrade() -> None:
    op.drop_constraint(CONSTRAINT_NAME, "reports", type_="unique")
