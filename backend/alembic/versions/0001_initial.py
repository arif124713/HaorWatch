"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-11-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import geoalchemy2

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")

    op.create_table(
        "study_areas",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("slug", sa.String(80), nullable=False, unique=True),
        sa.Column("name", sa.String(160), nullable=False),
        sa.Column("division", sa.String(80)),
        sa.Column("districts", postgresql.JSONB()),
        sa.Column("country_code", sa.String(2), nullable=False, server_default="BD"),
        sa.Column("boundary", geoalchemy2.types.Geometry(geometry_type="MULTIPOLYGON", srid=4326)),
        sa.Column("area_km2", sa.Numeric(12, 3)),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("metadata", postgresql.JSONB()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_study_areas_slug", "study_areas", ["slug"], unique=True)
    op.create_index(
        "ix_study_areas_boundary_gist",
        "study_areas",
        ["boundary"],
        postgresql_using="gist",
    )

    op.create_table(
        "dataset_assets",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "study_area_id",
            sa.Uuid(),
            sa.ForeignKey("study_areas.id", ondelete="SET NULL"),
        ),
        sa.Column("dataset_key", sa.String(80), nullable=False),
        sa.Column("display_name", sa.String(180)),
        sa.Column("provider", sa.String(120)),
        sa.Column("data_type", sa.String(40)),
        sa.Column("source_url", sa.Text()),
        sa.Column("source_version", sa.String(100)),
        sa.Column("acquired_at", sa.DateTime(timezone=True)),
        sa.Column("period_start", sa.Date()),
        sa.Column("period_end", sa.Date()),
        sa.Column("crs", sa.String(40)),
        sa.Column("resolution_m", sa.Numeric(10, 3)),
        sa.Column("license_note", sa.Text()),
        sa.Column("storage_uri", sa.Text()),
        sa.Column("checksum_sha256", sa.String(64)),
        sa.Column("metadata", postgresql.JSONB()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_dataset_assets_dataset_key", "dataset_assets", ["dataset_key"])

    op.create_table(
        "analysis_runs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "study_area_id",
            sa.Uuid(),
            sa.ForeignKey("study_areas.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", sa.String(24), nullable=False, server_default="queued"),
        sa.Column("analysis_label", sa.String(160), nullable=False),
        sa.Column("observation_start", sa.DateTime(timezone=True)),
        sa.Column("observation_end", sa.DateTime(timezone=True)),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("progress_percent", sa.SmallInteger()),
        sa.Column("progress_message", sa.String(240)),
        sa.Column("model_name", sa.String(120)),
        sa.Column("model_version", sa.String(80)),
        sa.Column("baseline_method", sa.String(80)),
        sa.Column("permanent_water_mask_applied", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("permanent_water_threshold", sa.Numeric(6, 3)),
        sa.Column("validation_status", sa.String(40), nullable=False, server_default="not_available"),
        sa.Column("validation_note", sa.Text()),
        sa.Column("processing_config", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("error_code", sa.String(80)),
        sa.Column("error_message_safe", sa.Text()),
        sa.Column("created_by", sa.String(80)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_analysis_runs_study_area_id", "analysis_runs", ["study_area_id"])
    op.create_check_constraint(
        "ck_percent_range",
        "analysis_runs",
        "progress_percent IS NULL OR (progress_percent >= 0 AND progress_percent <= 100)",
    )

    op.create_table(
        "analysis_summaries",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "analysis_run_id",
            sa.Uuid(),
            sa.ForeignKey("analysis_runs.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("study_area_km2", sa.Numeric(14, 4)),
        sa.Column("detected_flood_area_km2", sa.Numeric(14, 4)),
        sa.Column("baseline_flood_area_km2", sa.Numeric(14, 4)),
        sa.Column("mapped_agricultural_area_km2", sa.Numeric(14, 4)),
        sa.Column("potentially_exposed_agricultural_area_km2", sa.Numeric(14, 4)),
        sa.Column("potentially_exposed_agricultural_share_percent", sa.Numeric(7, 3)),
        sa.Column("valid_pixel_count", sa.BigInteger()),
        sa.Column("nodata_pixel_count", sa.BigInteger()),
        sa.Column("summary_metadata", postgresql.JSONB()),
    )

    op.create_table(
        "analysis_metrics",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "analysis_run_id",
            sa.Uuid(),
            sa.ForeignKey("analysis_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("method_key", sa.String(40), nullable=False),
        sa.Column("method_name", sa.String(120)),
        sa.Column("precision_score", sa.Numeric(8, 6)),
        sa.Column("recall_score", sa.Numeric(8, 6)),
        sa.Column("f1_score", sa.Numeric(8, 6)),
        sa.Column("iou_score", sa.Numeric(8, 6)),
        sa.Column("true_positive", sa.BigInteger()),
        sa.Column("false_positive", sa.BigInteger()),
        sa.Column("true_negative", sa.BigInteger()),
        sa.Column("false_negative", sa.BigInteger()),
        sa.Column("test_pixel_count", sa.BigInteger()),
        sa.Column(
            "test_dataset_asset_id",
            sa.Uuid(),
            sa.ForeignKey("dataset_assets.id", ondelete="SET NULL"),
        ),
        sa.Column("split_strategy", sa.String(160)),
        sa.Column("evaluated_at", sa.DateTime(timezone=True)),
        sa.Column("notes", sa.Text()),
        sa.UniqueConstraint("analysis_run_id", "method_key", name="uq_run_method"),
    )

    op.create_table(
        "analysis_layers",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "analysis_run_id",
            sa.Uuid(),
            sa.ForeignKey("analysis_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("layer_key", sa.String(60), nullable=False),
        sa.Column("display_name", sa.String(140)),
        sa.Column("available", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("geometry", geoalchemy2.types.Geometry(geometry_type="GEOMETRY", srid=4326)),
        sa.Column("storage_uri", sa.Text()),
        sa.Column("feature_count", sa.Integer()),
        sa.Column("area_km2", sa.Numeric(14, 4)),
        sa.Column("style", postgresql.JSONB()),
        sa.Column("metadata", postgresql.JSONB()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("analysis_run_id", "layer_key", name="uq_run_layer"),
    )
    op.create_index(
        "ix_analysis_layers_geometry_gist",
        "analysis_layers",
        ["geometry"],
        postgresql_using="gist",
    )

    op.create_table(
        "reports",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "analysis_run_id",
            sa.Uuid(),
            sa.ForeignKey("analysis_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("language", sa.String(8), nullable=False),
        sa.Column("generation_method", sa.String(24), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("title", sa.String(180), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("source_snapshot", postgresql.JSONB()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_reports_analysis_run_id", "reports", ["analysis_run_id"])


def downgrade() -> None:
    op.drop_table("reports")
    op.drop_table("analysis_layers")
    op.drop_table("analysis_metrics")
    op.drop_table("analysis_summaries")
    op.drop_table("analysis_runs")
    op.drop_table("dataset_assets")
    op.drop_table("study_areas")
