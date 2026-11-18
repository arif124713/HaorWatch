from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import StudyArea


class StudyAreaRepository:
    def __init__(self, db: Session):
        self.db = db

    def list_active(self) -> list[StudyArea]:
        return list(
            self.db.scalars(select(StudyArea).where(StudyArea.is_active.is_(True)).order_by(StudyArea.slug))
        )

    def get_by_slug(self, slug: str) -> StudyArea | None:
        return self.db.scalars(select(StudyArea).where(StudyArea.slug == slug)).first()

    def bounds_for(self, area: StudyArea) -> list[float] | None:
        if area.boundary is None:
            return None
        row = self.db.execute(
            select(
                func.ST_XMin(StudyArea.boundary),
                func.ST_YMin(StudyArea.boundary),
                func.ST_XMax(StudyArea.boundary),
                func.ST_YMax(StudyArea.boundary),
            ).where(StudyArea.id == area.id)
        ).one()
        if row is None or row[0] is None:
            return None
        return [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
