import json
from datetime import datetime, timezone

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.integrations.llm_client import LLMClient
from app.repositories.analysis import AnalysisRepository, utcnow
from app.schemas.report import ReportResponse

UNAVAILABLE_BN = "পাওয়া যায়নি"
UNAVAILABLE_EN = "Not available"

CAVEATS_BN = [
    "এই ফলাফল সম্ভাব্য কৃষিজমির সংস্পর্শ নির্দেশ করে; নিশ্চিত ফসলের ক্ষতি নয়।",
    "এটি বাস্তব সময়ের বন্যা সতর্কতা বা পূর্বাভাস নয়।",
]
CAVEATS_EN = [
    "These results indicate potentially exposed agricultural area; this is not confirmed crop damage.",
    "This is not a real-time flood warning or forecast.",
]

VALIDATION_NOTE_BN = "এই ফলাফলের কোনো স্বতন্ত্র বাহ্যিক যাচাই পাওয়া যায়নি।"
VALIDATION_NOTE_EN = "No independent external validation was available for these results."


class ReportService:
    def __init__(self, db=None, repo: AnalysisRepository | None = None, llm: LLMClient | None = None):
        self.repo = repo if repo is not None else AnalysisRepository(db)
        self.llm = llm if llm is not None else LLMClient(get_settings())

    def get_report(self, analysis_id, language: str) -> ReportResponse:
        if language not in ("bn", "en"):
            raise AppError("UNSUPPORTED_LANGUAGE", "Supported languages are bn and en.", 422)
        run = self.repo.get(analysis_id)
        if run is None:
            raise AppError("ANALYSIS_NOT_FOUND", "Analysis was not found.", 404)
        existing = self.repo.report_of(run.id, language)
        if existing is not None:
            return self._to_response(existing, run)
        row = self._generate(run, language)
        saved = self.repo.save_report(row) or row
        return self._to_response(saved, run)

    def _generate(self, run, language: str):
        snapshot = self._snapshot(run)
        body = None
        method = "template"
        if self.llm.enabled:
            text = self.llm.generate(self._prompt(snapshot, language))
            if text:
                body = text
                method = "llm"
        if body is None:
            body = self._template_body(snapshot, language)
            method = "template"
        from app.models import Report

        return Report(
            analysis_run_id=run.id,
            language=language,
            generation_method=method,
            status="completed",
            title=self._title(language),
            body=body,
            source_snapshot={"snapshot": snapshot, "generated_at": datetime.now(timezone.utc).isoformat()},
        )

    def _snapshot(self, run) -> dict:
        summary = self.repo.summary_of(run.id)
        return {
            "study_area_name": run.study_area.name if run.study_area else None,
            "analysis_id": str(run.id),
            "status": run.status,
            "observation_start": run.observation_start.isoformat() if run.observation_start else None,
            "observation_end": run.observation_end.isoformat() if run.observation_end else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "model_name": run.model_name,
            "model_version": run.model_version,
            "baseline_method": run.baseline_method,
            "permanent_water_mask_applied": run.permanent_water_mask_applied,
            "permanent_water_threshold": float(run.permanent_water_threshold)
            if run.permanent_water_threshold is not None
            else None,
            "validation_status": run.validation_status,
            "study_area_km2": float(summary.study_area_km2) if summary and summary.study_area_km2 is not None else None,
            "detected_flood_area_km2": float(summary.detected_flood_area_km2)
            if summary and summary.detected_flood_area_km2 is not None
            else None,
            "baseline_flood_area_km2": float(summary.baseline_flood_area_km2)
            if summary and summary.baseline_flood_area_km2 is not None
            else None,
            "mapped_agricultural_area_km2": float(summary.mapped_agricultural_area_km2)
            if summary and summary.mapped_agricultural_area_km2 is not None
            else None,
            "potentially_exposed_agricultural_area_km2": float(
                summary.potentially_exposed_agricultural_area_km2
            )
            if summary and summary.potentially_exposed_agricultural_area_km2 is not None
            else None,
            "potentially_exposed_agricultural_share_percent": float(
                summary.potentially_exposed_agricultural_share_percent
            )
            if summary and summary.potentially_exposed_agricultural_share_percent is not None
            else None,
        }

    def _title(self, language: str) -> str:
        if language == "bn":
            return "হাওর এলাকার বন্যা ও কৃষিজমির সম্ভাব্য ঝুঁকি"
        return "Flood and potential agricultural exposure in the haor area"

    def _template_body(self, snapshot: dict, language: str) -> str:
        if language == "bn":
            return self._template_body_bn(snapshot)
        return self._template_body_en(snapshot)

    def _template_body_bn(self, s: dict) -> str:
        area_name = s.get("study_area_name") or "নির্বাচিত হাওর এলাকা"
        lines = [
            f"এই রিপোর্টটি {area_name} এলাকার জন্য সংরক্ষিত বিশ্লেষণ ফলাফল থেকে তৈরি।",
            f"পর্যবেক্ষণ সময়কাল: {self._fmt_period(s, 'bn')}",
            f"সনাক্তকৃত বন্যা এলাকা (এমএল মডেল): {self._fmt_km2_bn(s.get('detected_flood_area_km2'))}।",
            f"বেসলাইন পদ্ধতিতে সনাক্তকৃত বন্যা এলাকা: {self._fmt_km2_bn(s.get('baseline_flood_area_km2'))}।",
            f"ম্যাপকৃত কৃষি এলাকা: {self._fmt_km2_bn(s.get('mapped_agricultural_area_km2'))}।",
            f"সম্ভাব্য কৃষি সংস্পর্শ এলাকা: {self._fmt_km2_bn(s.get('potentially_exposed_agricultural_area_km2'))} "
            f"(কৃষি এলাকার {self._fmt_pct_bn(s.get('potentially_exposed_agricultural_share_percent'))})।",
            f"মডেল: {s.get('model_name') or 'উপলব্ধ নয়'}, সংস্করণ: {s.get('model_version') or UNAVAILABLE_BN}।",
            f"স্থায়ী পানির মাস্ক: {'প্রয়োগ করা হয়েছে' if s.get('permanent_water_mask_applied') else 'প্রয়োগ করা হয়নি'}।",
        ]
        if s.get("validation_status") == "not_available":
            lines.append(VALIDATION_NOTE_BN)
        return "\n".join(lines)

    def _template_body_en(self, s: dict) -> str:
        area_name = s.get("study_area_name") or "the selected haor area"
        lines = [
            f"This report is generated from stored analysis results for {area_name}.",
            f"Observation period: {self._fmt_period(s, 'en')}.",
            f"Detected flood extent (ML model): {self._fmt_km2_en(s.get('detected_flood_area_km2'))}.",
            f"Detected flood extent (baseline): {self._fmt_km2_en(s.get('baseline_flood_area_km2'))}.",
            f"Mapped agricultural area: {self._fmt_km2_en(s.get('mapped_agricultural_area_km2'))}.",
            f"Potentially exposed agricultural area: {self._fmt_km2_en(s.get('potentially_exposed_agricultural_area_km2'))} "
            f"({self._fmt_pct_en(s.get('potentially_exposed_agricultural_share_percent'))} of mapped agricultural area).",
            f"Model: {s.get('model_name') or UNAVAILABLE_EN}, version: {s.get('model_version') or UNAVAILABLE_EN}.",
            f"Permanent water mask: {'applied' if s.get('permanent_water_mask_applied') else 'not applied'}.",
        ]
        if s.get("validation_status") == "not_available":
            lines.append(VALIDATION_NOTE_EN)
        return "\n".join(lines)

    def _fmt_period(self, s: dict, language: str) -> str:
        start = s.get("observation_start")
        end = s.get("observation_end")
        if start and end:
            return f"{str(start)[:10]} to {str(end)[:10]}"
        if start or end:
            return str(start or end)[:10]
        return UNAVAILABLE_BN if language == "bn" else UNAVAILABLE_EN

    def _fmt_km2_bn(self, value) -> str:
        if value is None:
            return UNAVAILABLE_BN
        return f"{value:,.2f} বর্গ কিলোমিটার"

    def _fmt_km2_en(self, value) -> str:
        if value is None:
            return UNAVAILABLE_EN
        return f"{value:,.2f} km2"

    def _fmt_pct_bn(self, value) -> str:
        if value is None:
            return UNAVAILABLE_BN
        return f"{value:,.2f}%"

    def _fmt_pct_en(self, value) -> str:
        if value is None:
            return UNAVAILABLE_EN
        return f"{value:,.2f}%"

    def _prompt(self, snapshot: dict, language: str) -> str:
        lang_name = "Bangla (বাংলা)" if language == "bn" else "English"
        return (
            "You write short flood-monitoring report summaries. "
            "Use ONLY the numbers in the JSON below; never invent, estimate, or forecast any value. "
            "Never claim confirmed crop damage, real-time warnings, or predictions. "
            "Describe agricultural overlap as potentially exposed agricultural area. "
            "If a value is null, say it is not available. Keep it under 200 words. "
            f"Write the report in {lang_name}.\n\nData:\n{json.dumps(snapshot, ensure_ascii=False)}"
        )

    def _caveats(self, language: str, run) -> list[str]:
        caveats = list(CAVEATS_BN if language == "bn" else CAVEATS_EN)
        if run.validation_status == "not_available":
            caveats.append(VALIDATION_NOTE_BN if language == "bn" else VALIDATION_NOTE_EN)
        return caveats

    def _to_response(self, row, run) -> ReportResponse:
        return ReportResponse(
            analysis_id=row.analysis_run_id,
            language=row.language,
            generation_method=row.generation_method,
            status=row.status,
            title=row.title,
            body=row.body,
            created_at=row.created_at or datetime.now(timezone.utc),
            data_as_of=run.completed_at or run.observation_end,
            caveats=self._caveats(row.language, run),
        )
