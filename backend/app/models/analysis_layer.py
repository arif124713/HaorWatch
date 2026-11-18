from decimal import Decimal
import uuid

from geoalchemy2 import Geometry
from sqlalchemy import Boolean, ForeignKey, Integer, Numeric, String, Text, Uuid, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, CreatedAtMixin, UUIDPrimaryKeyMixin


class AnalysisLayer(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "analysis_layers"
    __table_args__ = (UniqueConstraint("analysis_run_id", "layer_key", name="uq_run_layer"),)

    analysis_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False
    )
    layer_key: Mapped[str] = mapped_column(String(60), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(140))
    available: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    geometry = mapped_column(Geometry(geometry_type="GEOMETRY", srid=4326), nullable=True)
    storage_uri: Mapped[str | None] = mapped_column(Text)
    feature_count: Mapped[int | None] = mapped_column(Integer)
    area_km2: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    style: Mapped[dict] = mapped_column(JSONB, default=dict)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
