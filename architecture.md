# Architecture & Tech Stack — HaorWatch AI

## 1. System boundaries
Four layers:
1. **Data:** Sentinel-1 SAR, public labelled flood dataset, JRC Global Surface Water, ESA WorldCover, optional Sentinel-2 and external flood reference maps.
2. **Processing/ML:** preprocessing, permanent-water masking, baseline, custom ML inference, exposure overlay, evaluation, report context.
3. **Backend:** FastAPI serves study-area, analysis, map-layer, metrics, provenance, and report endpoints.
4. **Frontend:** React renders dashboard, map, evaluation, methodology, and reports.

Frontend does not perform raster processing or ML inference.

## 2. Exact MVP stack
### Backend
- Python 3.11+
- FastAPI, Uvicorn, Pydantic v2
- SQLAlchemy 2.x, Alembic
- PostgreSQL 16 + PostGIS
- GeoPandas, Shapely, Rasterio, PyProj, NumPy, SciPy
- Scikit-learn; PyTorch only if selected model requires it
- pytest

### Frontend
- React 18+, Vite, TypeScript strict mode
- Tailwind CSS, React Router
- TanStack Query for server state
- Leaflet + React-Leaflet (MapLibre acceptable alternative)
- Recharts optional; Lucide React icons

### Storage
- PostgreSQL/PostGIS for metadata, summaries, and vector geometries.
- Local filesystem or object storage for rasters/model artifacts.
- API returns GeoJSON for MVP vector layers. Do not send raw GeoTIFFs to browser.
- Never expose internal filesystem paths.

## 3. Data flow
1. Trigger or load a run for configured study area/time window.
2. Resolve source imagery and ancillary datasets, or load a prepared bundle.
3. Align rasters to common CRS, transform, resolution, extent, and nodata handling.
4. Apply permanent-water mask where available.
5. Generate Otsu/change-detection baseline.
6. Generate custom ML output independently.
7. Evaluate both against same held-out labels when available.
8. Vectorize/simplify map outputs.
9. Intersect flood extent with agricultural land-cover.
10. Persist summaries, metrics, map layers, provenance, and report.
11. Frontend retrieves completed run.

## 4. ML/geospatial implementation rules

### Sentinel-1 inputs
- Record product ID, acquisition date, polarization(s), orbit direction, and preprocessing.
- Use consistent VV/VH feature definitions across comparable inputs.
- Apply and document orbit correction, border/thermal noise handling, calibration, terrain correction, and speckle filtering choices as applicable.
- Align rasters before pixel-wise comparisons.
- Handle nodata explicitly; do not silently mix dB and linear units.
- **SAR units are explicit and deterministic.** The band unit (`dB` or `linear`) is declared in the model bundle `config.json` (`s1_unit`) and in dataset/inference configuration. Units are **never** inferred from pixel medians. Missing/unknown units raise a clear error, and a bundle whose `config.json` and `norm_stats.json` disagree is rejected. Training, validation, and inference use the same declared unit and identical preprocessing, and dB conversion is applied exactly once. Prepared analysis bundles must declare `s1_unit` in `manifest.json`.
- **Observation semantics are explicit.** An analysis records whether the Sentinel-1 input is a single acquisition or a temporal composite, the compositing method (mean), value domain (dB), image count, acquisition dates, orbit direction, polarisation, and product/scene IDs. A multi-date mean composite is not a single-date observation and does not represent peak flood extent.
- **GEE export tiling is projected-CRS based.** Export tiles are computed in the target projected CRS so each tile covers a constant ground area at any latitude; degree-based tiling would shrink ground coverage with latitude. Multi-tile mosaics reject inconsistent tile CRS.

### Common analysis grid
- `app/geospatial/grid.py` defines one authoritative `AnalysisGrid` (CRS, affine transform, width, height, pixel resolution, nodata) per run.
- Preprocessed SAR, U-Net output, Otsu baseline, JRC, and WorldCover are aligned to that exact grid. Alignment is validated on CRS, transform, dimensions, and pixel origin **before** any pixel-wise operation; `GridMismatchError` is raised on incompatibility. Matching array shapes alone is never treated as alignment.
- Categorical rasters (WorldCover class IDs, JRC) use nearest-neighbour resampling; continuous SAR bands use an explicit continuous method (default bilinear). The choice is always explicit, never implicit.
- Source NaN and declared source nodata are excluded from resampling and can never become valid zero-valued observations; destination pixels with no valid source keep the destination nodata.
- Rotated/sheared transforms are rejected; source arrays must match their declared grid dimensions.

### Background job execution limits
- Analysis runs execute in a subprocess launched from a FastAPI `BackgroundTasks` callback. This is sufficient for the MVP but has known limitations for long-running production use: the task runs in the API process, there is no queue, no retry, no horizontal scaling, and if the API process is killed mid-run the run can remain `processing` with no automatic recovery. A production deployment should move to a real job runner. This is documented rather than replaced to avoid adding uncontrolled infrastructure.
- Worker-persisted subprocess output (stdout/stderr) is truncated and passed through a redaction filter that strips likely secrets (API keys, tokens, passwords, operator keys, `Authorization: Bearer/Basic`, credential URL query parameters) before it is stored in `processing_config`.
- The worker records a sanitized subprocess outcome block on the run under `processing_config.subprocess`: `status` (completed/failed/timeout/launch_failed), `exit_code`, `timed_out`, `launched`, `failure_code`, `finished_at`, and bounded `stdout_excerpt`/`stderr_excerpt` (each capped at 2000 characters, redacted then truncated).
- Only a `queued` run is executed; `processing`/`completed`/`failed` runs are never re-executed. A late worker outcome never overwrites a run the pipeline already finalised (`completed`/`failed`). Database bookkeeping failures are logged but do not raise into the request that launched the worker.

### Reports
- Generated reports are persisted per `(analysis_run_id, language)`; a subsequent request for the same language reuses the stored report rather than regenerating it. Failed LLM generation falls back to the deterministic template (labelled `template`), and a report is never stored as `completed` unless a body was produced.
- The `reports` table has a database-level unique constraint `uq_report_run_language` on `(analysis_run_id, language)` (migration `0002_report_run_language_unique`). This prevents duplicate reports when concurrent requests generate the same report.
- Report generation is race-safe: on a unique-constraint conflict the service rolls back the failed transaction, re-reads the report persisted by the winning request, and returns it. If no report can be found after the conflict, or persistence fails for another reason, a `REPORT_PERSISTENCE_FAILED` error is raised rather than returning an unsaved report.

### Permanent-water mask
- Use JRC Global Surface Water with documented product/version and chosen occurrence/seasonality rule.
- Record threshold and mask status per run.
- Do not assume all haor water is permanent.
- If missing or unsuitable, set `permanent_water_mask_applied=false` and surface the caveat.

### Baseline
- Implement Otsu thresholding on documented SAR feature, or a reproducible before/after change-detection rule.
- Store baseline output separately from ML output.
- Record bands, threshold, filtering, and configuration.
- Do not tune on held-out test set.

### Custom model — U-Net is the MVP primary model
- **Required model:** train a custom U-Net semantic-segmentation model for pixel-wise flood/non-flood classification. Random Forest is optional experimentation, not the default MVP model.
- **Primary dataset candidate:** Sen1Floods11. Before training, audit its access terms/license, label definitions, available VV/VH inputs, nodata conventions, spatial metadata, and official split/event information. Record the exact dataset release/version and source.
- **Label strategy:** distinguish hand-labelled masks from weak labels. Use weak labels for training only after checking their semantics and quality; prioritize hand-labelled samples for validation/testing where feasible. Never present weak labels as ground truth.
- **Leakage control:** split by flood event/geographic region where the dataset permits; do not randomly split adjacent chips from the same event across train and test. Keep a genuinely held-out test set untouched during tuning.
- **Input pipeline:** use calibrated, consistently preprocessed Sentinel-1 VV/VH channels (and additional temporal/change channels only when paired acquisitions and compatible labels exist). Keep units consistent (linear vs dB), align channels/masks, normalize using training-set statistics only, and preserve nodata masks.
- **Initial training configuration:** start with a conventional 2D U-Net; use 256- or 512-pixel tiles according to GPU memory, batch size chosen to fit, Adam/AdamW, learning-rate scheduling, and early stopping on validation loss/IoU. Use a flood-aware loss such as weighted BCE + Dice (or BCE + Dice loss) to address class imbalance. Treat these as starting defaults to be tuned and logged, not claimed optimal settings.
- **Compute budget:** plan experiments within the available GPU allowance (up to approximately 30 GPU-hours per week if that is the active quota). Prioritize one reproducible baseline U-Net run, validation, and error analysis before hyperparameter sweeps. Save checkpoints regularly and resume interrupted runs.
- **Evaluation:** report precision, recall, F1, IoU, and confusion matrix on held-out hand-labelled/reference data where valid. Compare U-Net and Otsu on the same held-out pixels and valid-data mask. Report results separately by event/region where sample size permits.
- **Artifacts and reproducibility:** save model checkpoint, architecture/config, input channel order, normalization statistics, label mapping, dataset references/version/license notes, split manifest, random seed, training logs, best epoch, code commit/hash, and model version.
- **Model bundle metadata status.** A loadable bundle requires `checkpoint.pt`, `config.json` (with `s1_unit`), and `norm_stats.json` (with `s1_unit`). Current bundles:
  - `0.1.0`, `0.2.0`, `0.3.0`: declare `s1_unit=dB` in both `config.json` and `norm_stats.json`.
  - `0.4.0`: a fine-tune of `0.1.0`. Its training used the `0.1.0` `norm_stats.json` (dB). The registered bundle initially shipped only `checkpoint.pt`/`config.json`/`metrics_eval.json` (no unit metadata). During staging validation the unit was established as `dB` from the base artifact and finetune script, and the bundle was completed with `norm_stats.json` (from `0.1.0`) and an explicit `s1_unit: "dB"` plus `s1_unit_provenance` in `config.json`. Model weights were not modified.
  - The `default.txt` pointer selects the active bundle. `resolve_model_dir()` verifies the selected bundle against the manifest before it is used; a missing or corrupt default is an error, not a silent fallback.
- **Model artifact delivery strategy.** Model weights are **not** stored in Git (`backend/models/` and `*.pt` are gitignored) and are not in Git LFS. The tracked reproducibility manifest `app/ml/bundles_manifest.json` (accessed via `app/ml/model_registry.py`) records, for each version: required filenames, `model_version`, `s1_unit`, band order, probability threshold, base model, and SHA-256 checksums of **`checkpoint.pt`, `config.json`, and `norm_stats.json`** (manifest schema version 2). A deployment obtains bundles from the project's model artifact store (or regenerates them from the documented Kaggle notebooks) and places them at `backend/models/unet/<version>/`.
  - **Runtime verification is fail-closed and now covers every required artifact.** `model_registry.verify_bundle(version, model_root)` checks that each required file exists, that all three checksums match, and that the on-disk `config.json`/`norm_stats.json` metadata agrees with the manifest on model version, SAR unit, band order, and probability threshold. `load_bundle()` calls this verification before loading, and `resolve_model_dir()` verifies the configured default bundle rather than silently falling back to another checkpoint. A corrupt configured default is an error (`BundleVerificationError`); a fallback to a different verified bundle requires the explicit, documented `HAORWATCH_ALLOW_MODEL_FALLBACK` opt-in.
  - `verify_bundle()` is **malformed-metadata safe**: a non-object manifest, a non-object `config.json`/`norm_stats.json`, corrupted JSON, a non-numeric/out-of-range threshold, a wrong-typed or empty band order, a missing model version, an unsupported SAR unit, malformed/non-finite norm-stats, or a malformed checksum all produce a **structured failure report** (`ok: false`) rather than a raw exception. Only expected parsing/type/value errors are caught; unexpected programming errors propagate. Fail-closed behaviour for the configured default is unchanged.
  - `tests/unit/test_bundle_metadata.py` asserts the manifest is well-formed and, when on-disk bundles are present, that their checksums and metadata match; it also tests missing/corrupt checkpoint/config/norm-stats, wrong version/unit/band-order/threshold, malformed metadata types (invalid threshold, NaN/Inf/out-of-range, wrong band-order type, empty band order, missing version, malformed/non-finite norm-stats, malformed checksum), and a missing/corrupt default bundle — all on synthetic bundles so these paths run even without real weights.
- **Legacy trainer note.** `kaggle/haorwatch_unet_training.py` is a standalone, historical Kaggle trainer that inferred the unit from a median heuristic (`assume_db`). It is not part of the runtime path and is preserved for reproducibility of the existing checkpoints; it is not used to produce or serve bundles and does not affect the runtime SAR-unit contract.
- **Inference:** return a flood probability raster where practical, plus a thresholded class mask; record the probability threshold and post-processing. Keep U-Net output separate from Otsu baseline.
- **Fallback rule:** if dataset audit reveals incompatible or insufficient labels, document the blocker and obtain an approved alternative labelled dataset or use the baseline for the demo. Do not claim U-Net was trained or validated unless the run and evidence exist.

### Pseudo-label validity (Sentinel-2 reference)
- The Hakaluki v0.4 dataset uses Sentinel-2 MNDWI>0 as an automatic (proxy) label. Invalid reference pixels (NaN/inf or numeric nodata sentinels such as `-9999`) and pixels outside the AOI or with invalid SAR must **never** be treated as non-water (0) labels.
- `app/ml/labels.py` implements the validity rule: `build_validity_mask` intersects AOI ∧ valid SAR ∧ valid Sentinel-2; `apply_validity` writes the ignore sentinel `-1` for invalid pixels. The dataset builder writes `pseudo_label.tif` (with `-1` for invalid), `valid_mask.tif`, and `s1_valid_mask.tif`. The Hakaluki v0.4 fine-tune loader (`kaggle/haorwatch_finetune_v04.py`, `PairDataset`) uses `valid = finite(SAR) & (label != -1)` and, when present, intersects with `valid_mask.tif`.
- **Reprojection preserves nodata/masks.** `app/geospatial/reprojection.py` reprojects values and validity separately (`reproject_with_validity`), reading source nodata and dataset masks, excluding NaN/inf and finite sentinels, and using **nearest-neighbour for categorical (binary water) labels** so interpolation never fabricates fractional classes. Outside-source destination pixels stay invalid. This is what makes the validity mask meaningful: a resampled value cannot be recovered as "valid" if its source was already invalid.
- **Cache-validity contract.** A cached dataset pair is reusable only when it contains **all four** required artifacts — `s1_grid.tif`, `pseudo_label.tif`, `valid_mask.tif`, and `s1_valid_mask.tif` — there is no smaller "critical subset" that can be satisfied while a required artifact is missing (`REQUIRED_PAIR_ARTIFACTS`, with `CRITICAL_PAIR_ARTIFACTS` now an exact alias). `app/ml/dataset_cache.py` additionally validates: every raster is readable (a corrupted TIFF is a stale reason, not an exception); every raster matches the expected grid on **CRS, transform, shape, and resolution**; the four rasters agree with each other on shape; the binary validity masks contain only `{0, 1}`; the pseudo-label declares the configured ignore value as its nodata and contains only `{ignore, 0, 1}`; and the SAR grid declares nodata and keeps invalid pixels non-finite. The per-pair manifest record must agree on grid, SAR unit, label ignore value, source dates, builder version, and recorded valid-pixel count. Any failure marks the pair stale. Pairs from before the validity-mask fix (which lack `valid_mask.tif`/`s1_valid_mask.tif`) are detected and reported/skipped unless `--rebuild-stale` is passed; only the identified stale pair is rebuilt, and nothing is deleted automatically.
- **Train/eval mask consistency.** The fine-tune loader and its evaluation path (`predict_full`) both apply the same validity mask (valid SAR ∧ `valid_mask.tif` ∧ `label != -1`); previously the holdout evaluation used only SAR validity. Invalid pixels are excluded from loss and metrics and never become true negatives or false negatives.
- The existing `0.4.0` checkpoint was trained before the validity-mask fix was introduced; it is **not** reinterpreted and does **not** benefit from the corrected labels. Retraining (which would use the corrected labels) is required to benefit and is **not** performed here.

### Reproducible pipeline (DVC)
- `dvc.yaml` (with `params.yaml` and `.dvcignore` at the repository root) describes the reproducible stages: bundle-manifest verification, the DB-free unit gate, data preparation, AOI fetch, the Hakaluki v0.4 dataset build, U-Net training, bundle registration, Hakaluki diagnostics, and the PostGIS integration gate. Large/external artifacts (rasters, weights) are declared with `cache: false` and are not committed.
- **Parameters are connected to runtime behaviour**, not merely declared. `app/ml/build_config.py` is the single validated configuration path for the dataset builder: `build.sar_unit`, `build.s2_cloud_max`, `build.pair_window_days`, `build.label_ignore_value`, and `build.rebuild_stale` are passed **explicitly** as CLI arguments by the DVC stage and validated (unit supported; cloud % in `[0, 100]`; non-negative window; ignore value numeric and not `0`/`1`) **before** any GEE request or output file is created. Boolean `build.rebuild_stale` is passed with DVC's conditional-flag form (present only when `true`). The builder records the **effective configuration** in both the dataset manifest and each per-pair record, so provenance reflects what actually ran. `tests/unit/test_build_config.py` verifies each parameter changes the runtime config, that invalid values are rejected, and that the DVC stage declares and passes the same parameters.
- DVC reproducibility is verified at the parameter-to-runtime level only. A full end-to-end `dvc repro` of the GEE-backed stages requires Earth Engine credentials and dataset access and is **not** executed here. There is still **no published model artifact store**, so a fresh clone cannot obtain production weights without that source (BLOCKED).

### Evaluation
- Same held-out labelled samples and valid-pixel mask for ML and baseline.
- Positive class is flood.
- Metrics: precision, recall, F1, IoU; confusion matrix TP/FP/TN/FN.
- Record test count, class balance, split strategy, dataset, date.
- Prefer event/region-level reporting.
- No suitable ground truth means no accuracy claim.
- **Annotation-aware confusion matrix.** Positive labels, negative labels, and unannotated pixels are distinguished. Metrics are computed over explicitly annotated valid pixels only; unannotated pixels are excluded and never treated as background. Annotation polygons are never used to mask predictions in a way that hides false positives outside them.
- If annotations are positive-only, the result is recorded as positive-class agreement (`operator_annotated_positive_only`), not a full binary classification validation.

### Validation categories (kept distinct)
1. Operator-annotated reference validation (`operator_annotated`, `operator_annotated_positive_only`).
2. Optical satellite proxy comparison (`proxy_validated`, e.g. Sentinel-2 MNDWI) — not ground truth.
3. Independent external reference validation (`external_reference_checked`, e.g. Copernicus EMS/UNOSAT).
A weaker category is never presented as a stronger one. Record dataset source/version, label source/semantics, annotation and observation dates, evaluated pixel/area coverage, model checkpoint hash, git commit SHA, preprocessing configuration, split identifiers, and metric definitions.

### Observed U-Net / Otsu behaviour (staging diagnostics, not a performance claim)
- On the held-out Ghana Sen1Floods11 chip (`data/prepared/ghana-demo`, single 2018-09-18 acquisition), preprocessing was confirmed correct (VV/VH dB, band order `VV_dB,VH_dB`, matching normalization). U-Net (threshold 0.4) predicts a small water fraction (~1.2%) while Otsu (`VH_dB`, water-is-low, ~-14.8 dB) predicts ~42%. Their **agreement** with each other is low (IoU ~0.03) because they sit at opposite precision/recall operating points, not because of a preprocessing error. Against the chip's hand labels U-Net had F1 ~0.29 / IoU ~0.17 and Otsu F1 ~0.19 / IoU ~0.11 (positive class = water, ~5.7% of valid pixels). Neither agreement nor these values establish superiority; thresholds were **not** tuned.
- On the two Hakaluki operator tiles the operator labels are **positive-only**. Independent `scripts/hakaluki_diagnostics.py` run against bundle `0.4.0` (threshold 0.40, `s1_unit=dB`) reproduces and extends the earlier numbers: the two tiles do **not** overlap (adjacent projected bounds, no intersection); U-Net predicts a water fraction over valid pixels of ~0.704 and ~0.727; Otsu predicts ~0.582 and ~0.533. U-Net recall within the drawn polygons is 1.0 on both tiles (trivial, because over-detection cannot be penalised by positive-only labels); Otsu recall is ~0.949 and ~0.908. The U-Net probability histograms are strongly bimodal with ~62% of valid pixels near 1.0, corroborating systematic over-detection on Hakaluki SAR. This is positive-class agreement, not full binary accuracy; no independent external reference was used, and no threshold was tuned on these tiles.
- These diagnostics separate agreement from ground-truth accuracy and do not claim Hakaluki performance from Ghana data or vice versa.

### Exposure calculation
- Use ESA WorldCover class(es) from exact product legend/version. Class 40 is mapped cropland; it is a land-cover class, **not** a crop-specific (e.g. Boro rice) classification.
- Reproject land-cover and flood masks to a common grid.
- `potentially_exposed_agricultural_area_km2` is the area of intersection of detected flood and selected agricultural class after documented masking.
- This is "Mapped Cropland Potentially Exposed to Detected Water", not confirmed crop damage, Boro-specific area, yield loss, or financial loss.
- Exclude nodata pixels from area denominators and use consistent pixel-area calculations.
- Permanent-water masking is configurable; the occurrence threshold is recorded per run.

### Validation
- Preferred: date- and area-matched Copernicus EMS / UNOSAT flood reference.
- Alternative: suitable cloud-free Sentinel-2 water interpretation, clearly labelled reference proxy, not automatically ground truth.
- Record source, date, coverage, resolution, and limitations.
- Validation may be unavailable.

## 5. Backend folder structure
```text
backend/
  app/
    main.py
    core/{config.py,logging.py,exceptions.py}
    api/
      deps.py
      v1/{router.py,endpoints/}
    schemas/{common.py,study_area.py,analysis.py,layer.py,report.py}
    models/{base.py,study_area.py,dataset_asset.py,analysis_run.py,analysis_metric.py,report.py}
    repositories/
    services/{analysis_service.py,layer_service.py,report_service.py,provenance_service.py}
    geospatial/{raster_io.py,reprojection.py,grid.py,masks.py,vectorize.py,area.py,overlay.py}
    ml/{features.py,baseline.py,unet.py,dataset.py,train.py,inference.py,evaluate.py,model_registry.py}
    integrations/{gee_client.py,llm_client.py}
  scripts/{prepare_data.py,train_model.py,run_analysis.py,seed_demo.py}
  data/.gitkeep
  tests/{unit/,integration/}
  alembic/
  requirements.txt
  .env.example
  Dockerfile
```

## 6. Suggested frontend structure
```text
frontend/
  src/
    app/{App.tsx,router.tsx,providers.tsx}
    components/
      layout/{AppShell.tsx,Sidebar.tsx,Topbar.tsx}
      ui/{Card.tsx,Badge.tsx,Button.tsx,Skeleton.tsx,EmptyState.tsx,ErrorState.tsx}
      map/{FloodMap.tsx,MapLegend.tsx,LayerControls.tsx}
      metrics/{MetricCard.tsx,ModelComparison.tsx}
    pages/{OverviewPage.tsx,MapExplorerPage.tsx,ModelEvaluationPage.tsx,MethodologyPage.tsx,ReportPage.tsx}
    features/{study-area/,analysis/,reports/}
    lib/{api-client.ts,query-client.ts,formatters.ts,constants.ts}
    types/api.ts
    styles/index.css
  public/
  .env.example
```
Frontend may adjust structure but must preserve API contract and required screens.

## 7. Coding standards
- Small, single-purpose modules; explicit names; units in numeric field names.
- UTC ISO-8601 timestamps in API.
- No fake metrics or fake map geometry in production.
- Python: type hints, Pydantic I/O, SQLAlchemy models separate from schemas, dependency injection, structured logging, tests for masks/area/metrics/serialization.
- TypeScript: strict mode, avoid `any`, central API types, TanStack Query, explicit loading/error/empty states, no raster logic in components.
- Log request/correlation IDs; never expose stack traces.

## 8. Commands
### Backend
```bash
cd backend
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
pytest -q
alembic upgrade head
python -m scripts.seed_demo
```

### Migrations
- Migrations use Alembic (`backend/alembic/versions/`). Apply with `alembic upgrade head`; roll back one revision with `alembic downgrade -1`.
- `0002_report_run_language_unique` adds the `reports` unique constraint on `(analysis_run_id, language)`. Before adding the constraint it removes pre-existing duplicate report rows, keeping the earliest per `(analysis_run_id, language)` by `created_at, id` (deterministic; no non-report data is touched). The migration is reversible.
- Never edit an applied migration; add a new revision. Existing report rows remain readable.
- **Supported database:** PostgreSQL 16 or newer with PostGIS (for the geometry columns created in `0001_initial`).
- **Verification status (be precise):**
  - Full migration chain `0001_initial` → `0002_report_run_language_unique`: executed and **verified** on PostgreSQL 18.0 with PostGIS 3.6.2 (extension present; geometry columns `study_areas.boundary` MULTIPOLYGON SRID 4326 and `analysis_layers.geometry` GEOMETRY SRID 4326; GiST indexes present; `uq_report_run_language` present; downgrade to `0001_initial` and re-upgrade both succeed).
  - Report unique constraint (plain PostgreSQL): verified by the isolated-schema integration test.
  - `ReportService` concurrency over the full ORM schema: the test uses the real ORM models inside an isolated per-run schema (never the shared `public` schema), and synchronises both callers at the repository boundary so both complete the initial `report_of` lookup before either inserts. It asserts exactly one persisted report and identical returned report IDs. It requires a reachable PostgreSQL; in an environment without one it is **skipped**, and a verified run on a migrated PostGIS database is **not** claimed here for this environment.
  - Earlier isolated runs of migration `0002` core SQL on a plain `reports` table are superseded by the full-chain verification above.

### Database integration tests
- Unit tests run without a database. Any test that needs PostgreSQL is skipped unless `HAORWATCH_TEST_DATABASE_URL` points at a reachable, recognisably test database. Production-like names (including `haorwatch`) are refused unless explicitly allow-listed; unknown/ambiguous names require a deliberate override, e.g.:
  ```bash
  HAORWATCH_TEST_DATABASE_URL=postgresql+psycopg://postgres:postgres@127.0.0.1:5432/haorwatch_test \
  HAORWATCH_TEST_DB_ALLOWLIST=haorwatch_test pytest -q
  ```
- `tests/integration/_test_db.py` centralises the database-identity guard (allow-list `HAORWATCH_TEST_DB_ALLOWLIST`, deliberate override `HAORWATCH_ALLOW_NON_TEST_DB=1`, protected markers `prod`/`production`/`live`/`staging`/`haorwatch`); credentials are never logged.
- `tests/integration/test_report_uniqueness_db.py` creates a uniquely-named schema, pins `search_path`, creates minimal test tables inside it, and drops only that schema; it verifies the real unique constraint and a real concurrent-insert race without touching `public`.
- `tests/integration/test_report_service_concurrency_db.py` creates the **real ORM tables** in an isolated schema and exercises the real `ReportService` conflict path with a deterministic LLM double. The PostGIS extension schema is **discovered** from `pg_extension`/`pg_namespace` (it is not assumed to be `public`); each connection sets `search_path` to `<isolated schema>, <extension schema(s)>, public`, so the geometry type resolves via the extension schema while unqualified application tables resolve to the isolated schema first. The test confirms the ORM tables were created inside the isolated schema and drops only that schema; the PostGIS extension is never created, dropped, or modified.
### Frontend
```bash
cd frontend
npm install
npm run dev
npm run build
npm run lint
npm run typecheck
```
### Database
```bash
docker compose up -d db
docker compose down
```

## 9. Parallel integration agreement
- Backend owns canonical API contract.
- Frontend base URL: `VITE_API_BASE_URL`; API prefix `/api/v1`.
- CORS allows configured Vite origin only.
- Frontend may use exact-shape fixtures until endpoints are ready.
- If a contract changes, update `api.md` and TypeScript types and notify both developers.
