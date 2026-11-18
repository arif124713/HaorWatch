from datetime import datetime
from decimal import Decimal
import uuid

from sqlalchemy import BigInteger, DateTime, ForeignKey, Numeric, String, Text, Uuid, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDPrimaryKeyMixin


class AnalysisMetric(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "analysis_metrics"
    __table_args__ = (UniqueConstraint("analysis_run_id", "method_key", name="uq_run_method"),)

    analysis_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False
    )
    method_key: Mapped[str] = mapped_column(String(40), nullable=False)
    method_name: Mapped[str | None] = mapped_column(String(120))
    precision_score: Mapped[Decimal | None] = mapped_column(Numeric(8, 6))
    recall_score: Mapped[Decimal | None] = mapped_column(Numeric(8, 6))
    f1_score: Mapped[Decimal | None] = mapped_column(Numeric(8, 6))
    iou_score: Mapped[Decimal | None] = mapped_column(Numeric(8, 6))
    true_positive: Mapped[int | None] = mapped_column(BigInteger)
    false_positive: Mapped[int | None] = mapped_column(BigInteger)
    true_negative: Mapped[int | None] = mapped_column(BigInteger)
    false_negative: Mapped[int | None] = mapped_column(BigInteger)
    test_pixel_count: Mapped[int | None] = mapped_column(BigInteger)
    test_dataset_asset_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("dataset_assets.id", ondelete="SET NULL")
    )
    split_strategy: Mapped[str | None] = mapped_column(String(160))
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    notes: Mapped[str | None] = mapped_column(Text)
