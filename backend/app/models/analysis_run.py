from datetime import datetime
import uuid
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Numeric,
    SmallInteger,
    String,
    Text,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class AnalysisRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "analysis_runs"

    study_area_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("study_areas.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(24), default="queued", nullable=False)
    analysis_label: Mapped[str] = mapped_column(String(160), nullable=False)
    observation_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    observation_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    progress_percent: Mapped[int | None] = mapped_column(SmallInteger)
    progress_message: Mapped[str | None] = mapped_column(String(240))
    model_name: Mapped[str | None] = mapped_column(String(120))
    model_version: Mapped[str | None] = mapped_column(String(80))
    baseline_method: Mapped[str | None] = mapped_column(String(80))
    permanent_water_mask_applied: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    permanent_water_threshold: Mapped[Decimal | None] = mapped_column(Numeric(6, 3))
    validation_status: Mapped[str] = mapped_column(String(40), default="not_available", nullable=False)
    validation_note: Mapped[str | None] = mapped_column(Text)
    processing_config: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(80))
    error_message_safe: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str | None] = mapped_column(String(80))

    study_area = relationship("StudyArea", lazy="joined")
    summary = relationship(
        "AnalysisSummary", uselist=False, cascade="all, delete-orphan", lazy="selectin"
    )
    metrics = relationship("AnalysisMetric", cascade="all, delete-orphan", lazy="selectin")
    layers = relationship("AnalysisLayer", cascade="all, delete-orphan", lazy="selectin")
    reports = relationship("Report", cascade="all, delete-orphan", lazy="selectin")
