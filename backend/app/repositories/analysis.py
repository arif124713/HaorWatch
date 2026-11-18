from datetime import datetime, timezone
import uuid

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.models import AnalysisLayer, AnalysisMetric, AnalysisRun, AnalysisSummary, DatasetAsset, Report


class AnalysisRepository:
    def __init__(self, db: Session):
        self.db = db

    def latest_for_study_area(self, study_area_id: uuid.UUID) -> AnalysisRun | None:
        return self.db.scalars(
            select(AnalysisRun)
            .where(AnalysisRun.study_area_id == study_area_id)
            .order_by(AnalysisRun.created_at.desc())
        ).first()

    def list_for_study_area(
        self, study_area_id: uuid.UUID, limit: int
    ) -> tuple[list[AnalysisRun], int]:
        total = self.db.scalar(
            select(func.count()).select_from(AnalysisRun).where(AnalysisRun.study_area_id == study_area_id)
        )
        items = list(
            self.db.scalars(
                select(AnalysisRun)
                .where(AnalysisRun.study_area_id == study_area_id)
                .order_by(AnalysisRun.created_at.desc())
                .limit(limit)
            )
        )
        return items, int(total or 0)

    def get(self, run_id: uuid.UUID) -> AnalysisRun | None:
        return self.db.scalars(select(AnalysisRun).where(AnalysisRun.id == run_id)).first()

    def summary_of(self, run_id: uuid.UUID) -> AnalysisSummary | None:
        return self.db.scalars(select(AnalysisSummary).where(AnalysisSummary.analysis_run_id == run_id)).first()

    def layers_of(self, run_id: uuid.UUID) -> list[AnalysisLayer]:
        return list(self.db.scalars(select(AnalysisLayer).where(AnalysisLayer.analysis_run_id == run_id)))

    def layer(self, run_id: uuid.UUID, layer_key: str) -> AnalysisLayer | None:
        return self.db.scalars(
            select(AnalysisLayer)
            .where(AnalysisLayer.analysis_run_id == run_id)
            .where(AnalysisLayer.layer_key == layer_key)
        ).first()

    def metrics_of(self, run_id: uuid.UUID) -> list[AnalysisMetric]:
        return list(self.db.scalars(select(AnalysisMetric).where(AnalysisMetric.analysis_run_id == run_id)))

    def report_of(self, run_id: uuid.UUID, language: str) -> Report | None:
        return self.db.scalars(
            select(Report)
            .where(Report.analysis_run_id == run_id)
            .where(Report.language == language)
            .order_by(Report.created_at.desc())
        ).first()

    def save_report(self, report: Report) -> Report:
        self.db.add(report)
        self.db.commit()
        self.db.refresh(report)
        return report

    def dataset_assets_for(self, study_area_id: uuid.UUID) -> list[DatasetAsset]:
        return list(
            self.db.scalars(select(DatasetAsset).where(DatasetAsset.study_area_id == study_area_id))
        )

    def dataset_asset(self, asset_id: uuid.UUID) -> DatasetAsset | None:
        return self.db.scalars(select(DatasetAsset).where(DatasetAsset.id == asset_id)).first()

    def create_run(self, **fields) -> AnalysisRun:
        run = AnalysisRun(**fields)
        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)
        return run

    def update_run(self, run: AnalysisRun, **fields) -> AnalysisRun:
        for key, value in fields.items():
            setattr(run, key, value)
        self.db.commit()
        self.db.refresh(run)
        return run

    def next_queued_run(self) -> AnalysisRun | None:
        return self.db.scalars(
            select(AnalysisRun).where(AnalysisRun.status == "queued").order_by(AnalysisRun.created_at)
        ).first()

    def upsert_summary(self, summary: AnalysisSummary) -> AnalysisSummary:
        existing = self.summary_of(summary.analysis_run_id)
        if existing is not None:
            self.db.delete(existing)
            self.db.flush()
        self.db.add(summary)
        self.db.commit()
        self.db.refresh(summary)
        return summary

    def upsert_layer(self, layer: AnalysisLayer) -> AnalysisLayer:
        existing = self.layer(layer.analysis_run_id, layer.layer_key)
        if existing is not None:
            self.db.delete(existing)
            self.db.flush()
        self.db.add(layer)
        self.db.commit()
        self.db.refresh(layer)
        return layer

    def replace_metrics(self, run_id: uuid.UUID, metrics: list[AnalysisMetric]) -> list[AnalysisMetric]:
        self.db.execute(delete(AnalysisMetric).where(AnalysisMetric.analysis_run_id == run_id))
        for metric in metrics:
            self.db.add(metric)
        self.db.commit()
        return self.metrics_of(run_id)

    def upsert_dataset_asset(self, asset: DatasetAsset) -> DatasetAsset:
        self.db.add(asset)
        self.db.commit()
        self.db.refresh(asset)
        return asset


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
