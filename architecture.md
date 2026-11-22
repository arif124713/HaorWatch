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
- Worker-persisted subprocess output (stdout/stderr) is truncated and passed through a redaction filter that strips likely secrets (API keys, tokens, passwords, operator keys) before it is stored in `processing_config`.

### Reports
- Generated reports are persisted per `(analysis_run_id, language)`; a subsequent request for the same language reuses the stored report rather than regenerating it. Failed LLM generation falls back to the deterministic template (labelled `template`), and a report is never stored as `completed` unless a body was produced.

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
- **Inference:** return a flood probability raster where practical, plus a thresholded class mask; record the probability threshold and post-processing. Keep U-Net output separate from Otsu baseline.
- **Fallback rule:** if dataset audit reveals incompatible or insufficient labels, document the blocker and obtain an approved alternative labelled dataset or use the baseline for the demo. Do not claim U-Net was trained or validated unless the run and evidence exist.

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
