from datetime import date, datetime
import uuid

from sqlalchemy import CHAR, Date, DateTime, ForeignKey, Numeric, String, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, CreatedAtMixin, UUIDPrimaryKeyMixin


class DatasetAsset(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "dataset_assets"

    study_area_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("study_areas.id", ondelete="SET NULL")
    )
    dataset_key: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(180))
    provider: Mapped[str | None] = mapped_column(String(120))
    data_type: Mapped[str | None] = mapped_column(String(40))
    source_url: Mapped[str | None] = mapped_column(Text)
    source_version: Mapped[str | None] = mapped_column(String(100))
    acquired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    period_start: Mapped[date | None] = mapped_column(Date)
    period_end: Mapped[date | None] = mapped_column(Date)
    crs: Mapped[str | None] = mapped_column(String(40))
    resolution_m: Mapped[float | None] = mapped_column(Numeric(10, 3))
    license_note: Mapped[str | None] = mapped_column(Text)
    storage_uri: Mapped[str | None] = mapped_column(Text)
    checksum_sha256: Mapped[str | None] = mapped_column(CHAR(64))
    metadata_json: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
