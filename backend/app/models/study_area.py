from geoalchemy2 import Geometry
from sqlalchemy import Boolean, Numeric, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class StudyArea(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "study_areas"

    slug: Mapped[str] = mapped_column(String(80), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    division: Mapped[str | None] = mapped_column(String(80))
    districts: Mapped[list | None] = mapped_column(JSONB)
    country_code: Mapped[str] = mapped_column(String(2), default="BD", nullable=False)
    boundary = mapped_column(Geometry(geometry_type="MULTIPOLYGON", srid=4326), nullable=True)
    area_km2: Mapped[float | None] = mapped_column(Numeric(12, 3))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
