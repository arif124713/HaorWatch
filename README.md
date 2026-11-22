# HaorWatch AI — Backend & ML Developer Package

This package is specifically for the **backend/ML workstream**.

## Your responsibility
Build the data ingestion and processing pipeline, train the required U-Net flood-segmentation model, implement the separate Otsu baseline, complete geospatial analysis, persistence layer, and FastAPI service described in the specifications.

## Critical instruction: do not build or design the frontend
- **Do not create any frontend application, React components, pages, CSS, Tailwind styles, visual mockups, wireframes, or UI layouts.**
- **Do not make frontend design decisions** such as colors, typography, spacing, navigation, dashboard layout, animations, or visual identity.
- Do not add a `design.md` or attempt to prescribe how the frontend should look.
- Your deliverable is the backend/ML implementation and its documented API behavior.
- You may define API response fields and backend states required by the product, but do not implement their visual presentation.

## Source of truth
Read in this order: `prd.md`, `architecture.md`, `schema.md`, `api.md`, then `flows.md`.

- `api.md` is the API contract. Keep endpoint paths, field names, types, null semantics, and error conventions consistent.
- `schema.md` defines persistence entities and validation rules.
- `architecture.md` defines the intended processing and service boundaries.
- `prd.md` defines product scope and acceptance criteria.
- `flows.md` describes user/product behavior to support through backend capabilities; it is not an instruction to build UI.

## Non-negotiable product and data rules
- Never fabricate metrics, affected-area values, dates, validation results, or map geometries.
- Distinguish custom ML predictions, conventional baseline outputs, and reference data.
- Use “potentially exposed agricultural area”; never claim confirmed crop damage.
- Compare ML and baseline on the same held-out evaluation samples where valid labels permit.
- Record provenance, acquisition dates, processing configuration, model version, and evaluation metrics.
- Null means unavailable; zero means measured zero.
- Do not claim flood forecasting, emergency warnings, or verified crop-loss estimation.
- Confirm the pilot AOI geometry before producing spatial results.

## Expected backend deliverables
1. FastAPI service matching `api.md`.
2. PostgreSQL/PostGIS schema and Alembic migrations.
3. Reproducible Sentinel-1 preprocessing and geospatial pipeline.
4. Reproducibly trained U-Net model, documented Otsu baseline, and held-out evaluation workflow.
5. Persisted analysis summaries, layers, metrics, and source provenance.
6. Tests for API contracts, geospatial calculations, validation, and failure states.
7. `.env.example` and setup/run instructions.

Frontend integration should be possible using the API contract alone. Do not build the frontend yourself.

## Required model decision
U-Net is the primary custom model for this MVP. Sen1Floods11 is the primary dataset candidate and must be audited before training. Random Forest is optional, not a replacement for the U-Net requirement. Work within the available GPU quota (approximately 30 GPU-hours/week if applicable); prioritize a reproducible run, checkpointing, held-out evaluation, and error analysis over broad hyperparameter sweeps.

## Technical correctness & reliability notes
- **SAR units are explicit and deterministic.** Model bundles declare `s1_unit` (`dB`/`linear`); units are never inferred from pixel medians; missing units raise an error. Training and inference share one preprocessing definition.
- **One common analysis grid.** Preprocessed SAR, U-Net, Otsu, JRC, and WorldCover are aligned to a single `AnalysisGrid` (CRS, transform, dimensions, resolution, nodata) that is validated before pixel-wise operations.
- **Annotation-aware metrics.** Hand-label evaluation distinguishes positive/negative/unannotated pixels; unannotated pixels are never treated as background, and positive-only annotations are not claimed as full binary validation.
- **Observation semantics** (single vs temporal mean composite, image count, dates, orbit, polarisation, product IDs) are recorded and surfaced; composites are not described as single-date observations or peak flood extent.
- **Analysis jobs** cannot remain silently queued: a live run requires both observation dates (or an explicit `prepared_dir`), and the subprocess worker records exit code/stdout/stderr/timeout and marks failures visibly.
- **Validation categories** (operator annotation, optical proxy, independent external reference) are kept distinct; ESA WorldCover class 40 is "mapped cropland", and exposure is reported as "Mapped Cropland Potentially Exposed to Detected Water", never confirmed crop damage.
- API routes are registered once under `/api/v1`; each run executes once.

### Tests
```bash
cd backend
python -m pytest -q
```
Database-backed tests are skipped unless a reachable PostgreSQL is configured:
```bash
HAORWATCH_TEST_DATABASE_URL=postgresql+psycopg://postgres:postgres@127.0.0.1:5432/haorwatch python -m pytest -q
```

### Verification status
- Unit suite: implemented and passing without a database.
- Database integration: tests run against a real PostgreSQL and isolate their own schema (never drop public tables); require `HAORWATCH_TEST_DATABASE_URL`.
- Migrations: Alembic `0001_initial` → `0002_report_run_language_unique`; single head. The **full chain including PostGIS** (extension, geometry columns, GiST indexes) was executed and verified on PostgreSQL 18.0 + PostGIS 3.6.2, including downgrade/re-upgrade.
- Report `ReportService` concurrency over the full ORM schema: verified on the migrated PostGIS database.
- Model bundles: weights are not in Git; the tracked manifest `app/ml/model_registry.py::bundles_manifest.json` records filenames and SHA-256 checksums; every loadable bundle declares `s1_unit=dB`.
- Local real-data pipeline: validated on the documented prepared bundles with the real checkpoint (stages only; not the live GEE/Sentinel-1 acquisition path).
- Live GEE/Sentinel-1 acquisition: not verified here (requires external credentials/data).

