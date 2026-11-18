from decimal import Decimal
import uuid

from sqlalchemy import BigInteger, ForeignKey, Numeric, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDPrimaryKeyMixin


class AnalysisSummary(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "analysis_summaries"

    analysis_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("analysis_runs.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    study_area_km2: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    detected_flood_area_km2: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    baseline_flood_area_km2: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    mapped_agricultural_area_km2: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    potentially_exposed_agricultural_area_km2: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    potentially_exposed_agricultural_share_percent: Mapped[Decimal | None] = mapped_column(Numeric(7, 3))
    valid_pixel_count: Mapped[int | None] = mapped_column(BigInteger)
    nodata_pixel_count: Mapped[int | None] = mapped_column(BigInteger)
    summary_metadata: Mapped[dict] = mapped_column(JSONB, default=dict)
