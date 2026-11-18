# HaorWatch AI — Backend & ML

FastAPI service, PostGIS persistence, Sentinel-1 flood pipeline, Otsu baseline,
and the custom U-Net model workflow. No frontend lives here; the contract with
the frontend is `../api.md`.

## Setup (PowerShell)

The repo-root virtualenv is already created (`..\venv`). To use it:

```powershell
cd backend
..\venv\Scripts\Activate.ps1        # or reference ..\venv\Scripts\python.exe directly
pip install -r requirements.txt     # already done for ..\venv
```

PyTorch is only needed for inference/training tooling:

```powershell
pip install -r requirements-ml.txt
```

Database (PostgreSQL 16 + PostGIS via Docker):

```powershell
cd ..            # repo root
docker compose up -d db
cd backend
..\venv\Scripts\python.exe -m alembic upgrade head
..\venv\Scripts\python.exe -m scripts.seed_demo
```

`seed_demo` inserts the pilot study area `hakaluki-haor` with
`aoi_status=pending_confirmation` (boundary deliberately null until the exact
AOI geometry is confirmed) plus the source catalogue. Spatial results are
refused until the boundary exists.

## Run the API

```powershell
..\venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Endpoints follow `../api.md` under `/api/v1` (also mounted at root for
convenience): health, study areas, latest/list/detail analyses, layer
catalogue, GeoJSON layers, metrics, sources, reports (`bn`/`en`), and the
operator-only `POST /admin/analyses` (queued runs; execute them with
`scripts.run_analysis --run-id <id>`).

## Environment

Copy `.env.example` to `.env`. Key values:

- `DATABASE_URL` — PostGIS connection string.
- `OPERATOR_API_KEY` — replace with a long random secret before exposing the admin endpoint.
- `GOOGLE_EARTH_ENGINE_PROJECT` — required for live AOI exports; sign up at
  <https://earthengine.google.com/> and register the project.
- `LLM_PROVIDER` / `GEMINI_API_KEY` — optional; template reports are the default
  and fallback.

## Tests

```powershell
..\venv\Scripts\python.exe -m pytest tests
```

43 unit/contract tests cover the API envelope, geospatial math, Otsu baseline,
metric edge cases, schema validation bounds, and report templates. They run
without a database (DB integration tests would need the Docker service up).

## Model training — Kaggle only

Never train locally. See `kaggle/README.md`; the short version: import
`kaggle/train_haorwatch_unet.ipynb` into Kaggle with a GPU accelerator, run it,
download `haorwatch_unet_v0.1.0.zip`, then:

```powershell
..\venv\Scripts\python.exe -m scripts.train_model --register models\unet\haorwatch_unet_v0.1.0
```

## Analysis pipeline

Once the AOI boundary is confirmed and a model is registered:

```powershell
..\venv\Scripts\python.exe -m scripts.run_analysis --start 2026-05-01T00:00:00Z --end 2026-05-15T00:00:00Z
```

The pipeline exports Sentinel-1 VV/VH (dB), JRC occurrence (permanent water,
threshold 95), and ESA WorldCover v200 over the AOI at 10 m in the local UTM
zone, computes the Otsu baseline, runs U-Net inference, masks permanent water,
intersects flood with cropland class 40, vectorizes layers, persists
summaries/provenance, and stores a Bangla template report. Without a registered
checkpoint it completes with the baseline only and `ml_flood` marked
unavailable — never fabricated.

Prepared bundles (no GEE required): pass `--prepared-dir` containing
`s1.tif`, `jrc_occurrence.tif`, `worldcover.tif`, optional `manifest.json`.
The run is labelled prepared in `processing_config`.

## Layout

- `app/api/v1/endpoints/` — HTTP layer, thin controllers.
- `app/services/` — business logic; repositories injected, so contract tests run DB-free.
- `app/models/` + `alembic/` — PostGIS schema (`0001_initial`).
- `app/geospatial/` — raster IO, reprojection/UTM grids, masks, vectorize, area math.
- `app/ml/` — U-Net, dataset, Otsu baseline, metrics, inference, registry, local trainer.
- `app/integrations/` — GEE export client, optional Gemini report phrasing.
- `data/`, `models/` — runtime artifacts (gitignored, never exposed via API).
