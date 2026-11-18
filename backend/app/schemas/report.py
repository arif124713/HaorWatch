from datetime import datetime
import uuid

from pydantic import BaseModel


class ReportResponse(BaseModel):
    analysis_id: uuid.UUID
    language: str
    generation_method: str
    status: str
    title: str
    body: str
    created_at: datetime
    data_as_of: datetime | None = None
    caveats: list[str] = []
