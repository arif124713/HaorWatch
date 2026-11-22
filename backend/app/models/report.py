from datetime import datetime
import uuid

from sqlalchemy import DateTime, ForeignKey, String, Text, UniqueConstraint, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, CreatedAtMixin, UUIDPrimaryKeyMixin


class Report(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "reports"
    __table_args__ = (
        UniqueConstraint("analysis_run_id", "language", name="uq_report_run_language"),
    )

    analysis_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    language: Mapped[str] = mapped_column(String(8), nullable=False)
    generation_method: Mapped[str] = mapped_column(String(24), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    title: Mapped[str] = mapped_column(String(180), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    source_snapshot: Mapped[dict] = mapped_column(JSONB, default=dict)
