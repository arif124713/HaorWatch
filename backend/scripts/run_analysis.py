import argparse
import json
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import shapely
from geoalchemy2.shape import to_shape
from shapely.geometry import mapping
from shapely.ops import unary_union

from app.core.config import get_settings
from app.core.database import get_session_factory
from app.core.exceptions import AppError, ProviderUnavailableError
from app.geospatial.area import mask_area_km2, projected_area_km2
from app.geospatial.grid import (
    AnalysisGrid,
    GridMismatchError,
    align_to_grid,
    assert_grids_match,
    grid_from_profile,
    validate_many,
)
from app.geospatial.masks import apply_permanent_water, compute_valid
from app.geospatial.reprojection import grid_for_bounds, reproject_geometry, utm_epsg_for
from app.geospatial.raster_io import read_bands, read_single, write_raster
from app.geospatial.vectorize import ensure_multipolygon, mask_to_geometry, simplify_if_large
from app.integrations.gee_client import GEEClient
from app.ml.baseline import otsu_flood_mask
from app.ml.features import band_to_db, normalize_unit
from app.ml.inference import predict_raster
from app.ml.model_registry import resolve_model_dir
from app.repositories.analysis import AnalysisRepository
from app.repositories.study_area import StudyAreaRepository

# Permanent-water threshold and cropland classes are configurable via Settings
# (permanent_water_occurrence_threshold, agricultural_class_ids).
WORLDCOVER_SOURCE = {
    "dataset_key": "esa_worldcover",
    "display_name": "ESA WorldCover",
    "provider": "European Space Agency",
    "role": "Mapped cropland overlay (class 40)",
    "version": "v200 (2021)",
    "resolution_m": 10.0,
    "status": "used",
    "notes": "WorldCover class 40 is 'Cropland'. It is a land-cover class, not a crop-specific "
             "(e.g. Boro rice) classification, and is not confirmed cultivation or crop damage.",
}


def run_pipeline(study_area_slug: str, start: str, end: str, run_id: str | None = None,
                 model_dir: str | None = None, orbit: str | None = None,
                 prepared_dir: str | None = None) -> uuid.UUID:
    settings = get_settings()
    factory = get_session_factory()
    with factory() as db:
        repo = AnalysisRepository(db)
        area_repo = StudyAreaRepository(db)
        area = area_repo.get_by_slug(study_area_slug)
        if area is None:
            raise SystemExit(f"Study area '{study_area_slug}' not found. Run scripts.seed_demo first.")
        if area.boundary is None:
            raise SystemExit(
                "Study-area AOI geometry is not confirmed. Confirm the AOI before producing spatial results."
            )

        if run_id is not None:
            run = repo.get(uuid.UUID(run_id))
            if run is None:
                raise SystemExit(f"Analysis run {run_id} not found.")
        else:
            manifest = {}
            if prepared_dir:
                manifest_path = Path(prepared_dir) / "manifest.json"
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            obs_start = datetime.fromisoformat(start.replace("Z", "+00:00")) if start else None
            obs_end = datetime.fromisoformat(end.replace("Z", "+00:00")) if end else None
            if prepared_dir:
                obs_start = (
                    datetime.fromisoformat(manifest["observation_start"])
                    if manifest.get("observation_start")
                    else obs_start
                )
                obs_end = (
                    datetime.fromisoformat(manifest["observation_end"])
                    if manifest.get("observation_end")
                    else obs_end
                )
            label = manifest.get("label") if prepared_dir and manifest.get("label") else (
                f"Sentinel-1 analysis {start[:10]} to {end[:10]}" if start and end else "Prepared analysis"
            )
            run = repo.create_run(
                study_area_id=area.id,
                status="processing",
                analysis_label=label,
                observation_start=obs_start,
                observation_end=obs_end,
                started_at=datetime.now(timezone.utc),
                progress_percent=0,
                progress_message="Starting pipeline",
                created_by="pipeline",
                processing_config={},
            )

        run_dir = Path(settings.data_root) / "runs" / str(run.id)
        try:
            _execute(db, repo, area, run, run_dir, start, end, model_dir, orbit, prepared_dir, settings)
        except Exception as exc:
            db.rollback()
            safe_code = "PIPELINE_ERROR"
            message = "Analysis processing failed."
            if isinstance(exc, ProviderUnavailableError):
                safe_code = exc.code
                message = exc.message
            repo.update_run(
                run,
                status="failed",
                error_code=safe_code,
                error_message_safe=message,
                completed_at=datetime.now(timezone.utc),
                progress_message=None,
            )
            traceback.print_exc()
            print(f"Run marked failed: {safe_code}", file=sys.stderr)
            raise SystemExit(1)
        return run.id


def _execute(db, repo, area, run, run_dir: Path, start, end, model_dir, orbit, prepared_dir, settings):
    permanent_water_threshold = float(settings.permanent_water_occurrence_threshold)
    agricultural_classes = list(settings.agricultural_class_list)
    boundary = area.boundary if isinstance(area.boundary, shapely.geometry.base.BaseGeometry) else to_shape(area.boundary)
    geom_wgs84 = ensure_multipolygon(boundary)
    if geom_wgs84 is None or geom_wgs84.is_empty:
        raise SystemExit("Study-area boundary is empty.")

    lon, lat = geom_wgs84.centroid.x, geom_wgs84.centroid.y
    epsg = utm_epsg_for(lon, lat)
    utm_crs = f"EPSG:{epsg}"
    bounds = geom_wgs84.bounds
    transform, grid_shape = grid_for_bounds(bounds, epsg, 10.0)

    repo.update_run(run, progress_percent=10, progress_message="Resolving input imagery")

    sources = []
    manifest = {}
    observation = None
    if prepared_dir:
        s1_path = Path(prepared_dir) / "s1.tif"
        jrc_path = Path(prepared_dir) / "jrc_occurrence.tif"
        wc_path = Path(prepared_dir) / "worldcover.tif"
        manifest_path = Path(prepared_dir) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        sources = list(manifest.get("sources", []))
        observation = manifest.get("observation")
        prepared_note = "Prepared/precomputed input bundle."
        jrc_available = jrc_path.exists()
        wc_available = wc_path.exists()
        for key, meta in (
            ("jrc_gsw", {
                "dataset_key": "jrc_gsw",
                "display_name": "JRC Global Surface Water",
                "provider": "European Commission Joint Research Centre",
                "role": "Permanent-water exclusion mask",
                "version": "1.4",
                "status": "unavailable",
                "notes": "Not included in the prepared bundle; permanent-water masking skipped.",
            }),
            ("esa_worldcover", {
                "dataset_key": "esa_worldcover",
                "display_name": "ESA WorldCover",
                "provider": "European Space Agency",
                "role": "Agricultural land-cover overlay",
                "version": "v200 (2021)",
                "status": "unavailable",
                "notes": "Not included in the prepared bundle; agricultural exposure not computed.",
            }),
        ):
            if (key == "jrc_gsw" and not jrc_available) or (key == "esa_worldcover" and not wc_available):
                if not any(s.get("dataset_key") == key for s in sources):
                    sources.append(meta)
    else:
        client = GEEClient(settings.google_earth_engine_project)
        repo.update_run(run, progress_percent=20, progress_message="Exporting Sentinel-1 mosaic")
        s1_image, s1_meta = client.s1_mosaic(mapping(geom_wgs84), start, end, orbit)
        s1_path = client.export_image(
            s1_image, mapping(geom_wgs84), utm_crs, 10.0, run_dir / "s1.tif", band_names=["VV", "VH"]
        )
        sources.append({
            "dataset_key": "sentinel1_sar",
            "display_name": "Sentinel-1 SAR",
            "provider": "Copernicus",
            "role": "Flood detection input imagery",
            "version": s1_meta.get("collection"),
            "period_start": (start[:10] if start else None),
            "period_end": (end[:10] if end else None),
            "resolution_m": 10.0,
            "crs": utm_crs,
            "status": "used",
            "notes": json.dumps(s1_meta),
        })
        observation = {
            "type": s1_meta.get("observation_type", "temporal_composite"),
            "compositing_method": s1_meta.get("compositing_method"),
            "value_domain": s1_meta.get("value_domain"),
            "image_count": s1_meta.get("image_count"),
            "acquisition_dates": s1_meta.get("acquisition_dates"),
            "orbit_direction": s1_meta.get("orbit_direction"),
            "polarisation": s1_meta.get("polarisation"),
            "period_start": start,
            "period_end": end,
            "product_ids": s1_meta.get("product_ids"),
            "note": s1_meta.get("note"),
        }

        repo.update_run(run, progress_percent=40, progress_message="Exporting permanent-water and land-cover")
        jrc_image, jrc_meta = client.jrc_occurrence(mapping(geom_wgs84))
        jrc_path = client.export_image(jrc_image, mapping(geom_wgs84), utm_crs, 10.0, run_dir / "jrc_occurrence.tif")
        sources.append({
            "dataset_key": "jrc_gsw",
            "display_name": "JRC Global Surface Water",
            "provider": "European Commission Joint Research Centre",
            "role": "Permanent-water exclusion mask",
            **{k: v for k, v in jrc_meta.items() if k in ("version", "resolution_m", "crs", "source_url")},
            "status": "used",
            "notes": json.dumps(jrc_meta),
        })
        wc_image, wc_meta = client.worldcover(mapping(geom_wgs84))
        wc_path = client.export_image(wc_image, mapping(geom_wgs84), utm_crs, 10.0, run_dir / "worldcover.tif")
        sources.append({**WORLDCOVER_SOURCE, "notes": json.dumps(wc_meta)})
        prepared_note = None
        jrc_available = True
        wc_available = True

    repo.update_run(run, progress_percent=55, progress_message="Aligning rasters")

    # Authoritative analysis grid: UTM CRS, 10 m pixels, fixed origin/size for
    # the study-area bounds. All rasters are aligned to this exact grid.
    s1_data, s1_profile = read_bands(s1_path)
    target_grid = AnalysisGrid(
        crs=utm_crs,
        transform=transform,
        width=int(grid_shape[1]),
        height=int(grid_shape[0]),
        resolution_m=10.0,
        nodata=np.nan,
    )
    s1_grid = grid_from_profile(s1_profile, resolution_m=10.0, require_projected=False)
    vv = align_to_grid(s1_data[0], s1_grid, target_grid, categorical=False)
    vh = align_to_grid(s1_data[1], s1_grid, target_grid, categorical=False)
    s1_transform, s1_shape = target_grid.transform, target_grid.shape

    jrc_data = None
    if jrc_available:
        jrc_data, jrc_profile = read_single(jrc_path)
        jrc_grid = grid_from_profile(jrc_profile, resolution_m=10.0, require_projected=False)
        jrc_data = align_to_grid(jrc_data, jrc_grid, target_grid, categorical=True)
    wc_data = None
    if wc_available:
        wc_data, wc_profile = read_single(wc_path)
        wc_grid = grid_from_profile(wc_profile, resolution_m=10.0, require_projected=False)
        wc_data = align_to_grid(wc_data, wc_grid, target_grid, categorical=True)

    # Explicit alignment assertion before any pixel-wise operation.
    check_grids = {"sentinel1": target_grid}
    if jrc_data is not None:
        check_grids["jrc"] = AnalysisGrid(utm_crs, s1_transform, int(s1_shape[1]), int(s1_shape[0]), 10.0)
    validate_many(check_grids)
    if jrc_data is not None and jrc_data.shape != s1_shape:
        raise GridMismatchError("JRC raster is not aligned to the analysis grid.")
    if wc_data is not None and wc_data.shape != s1_shape:
        raise GridMismatchError("WorldCover raster is not aligned to the analysis grid.")

    valid = compute_valid(vv, vh)
    from rasterio.features import rasterize

    geom_utm = reproject_geometry(geom_wgs84, "EPSG:4326", utm_crs)
    aoi_mask = (
        rasterize(
            [(mapping(geom_utm), 1)],
            out_shape=s1_shape,
            transform=s1_transform,
            fill=0,
            dtype="uint8",
        ).astype(bool)
    )
    valid = valid & aoi_mask
    valid_count = int(valid.sum())
    total_pixels = int(vv.size)
    nodata_count = total_pixels - valid_count

    # Declared SAR unit is mandatory. Prepared bundles must declare it in
    # manifest.json; live GEE exports are dB backscatter by construction.
    if prepared_dir:
        if "s1_unit" not in manifest:
            raise SystemExit(
                "Prepared bundle manifest.json must declare 's1_unit' ('dB' or 'linear'); "
                "refusing to guess Sentinel-1 units."
            )
        s1_unit = normalize_unit(manifest["s1_unit"])
    else:
        s1_unit = "dB"
    if s1_unit != "dB" and s1_unit != "linear":
        raise SystemExit(f"Unsupported declared s1_unit {s1_unit!r}.")

    # Write the preprocessed SAR onto the analysis grid in its DECLARED unit.
    # U-Net inference and the Otsu baseline both consume this exact raster, and
    # both apply band_to_db() using the same declared unit, so dB conversion
    # happens exactly once.
    preprocessed_path = write_raster(
        run_dir / "s1_preprocessed.tif",
        np.stack([vv.astype(np.float32), vh.astype(np.float32)]),
        s1_transform,
        utm_crs,
        dtype="float32",
        nodata=np.nan,
    )

    repo.update_run(run, progress_percent=65, progress_message="Computing Otsu baseline")
    baseline_mask, baseline_threshold = otsu_flood_mask(band_to_db(vh, s1_unit), valid, water_is_low=True)
    baseline_area = mask_area_km2(baseline_mask, s1_transform)

    if model_dir is None:
        resolved = resolve_model_dir(settings.model_root)
        model_dir = str(resolved) if resolved else None
    ml_mask = None
    probability_threshold = None
    if model_dir:
        repo.update_run(run, progress_percent=75, progress_message="Running custom ML model")
        inference_result = predict_raster(
            model_dir, preprocessed_path, run_dir, s1_unit=s1_unit
        )
        probability_threshold = inference_result.get("threshold")
        ml_mask_data, ml_profile = read_single(inference_result["mask_path"])
        ml_grid = grid_from_profile(ml_profile, resolution_m=10.0)
        target_grid = AnalysisGrid(utm_crs, s1_transform, int(s1_shape[1]), int(s1_shape[0]), 10.0)
        assert_grids_match(target_grid, ml_grid, label="ml_inference_output")
        ml_mask = ml_mask_data > 0.5
        ml_mask &= valid

    mask_applied = jrc_data is not None and jrc_data.shape == s1_shape
    if ml_mask is not None:
        flood_mask = apply_permanent_water(ml_mask, jrc_data if mask_applied else None, permanent_water_threshold)
    else:
        flood_mask = None

    baseline_mask = apply_permanent_water(
        baseline_mask, jrc_data if mask_applied else None, permanent_water_threshold
    )
    detected_area = mask_area_km2(flood_mask, s1_transform) if flood_mask is not None else None

    repo.update_run(run, progress_percent=85, progress_message="Computing agricultural exposure")
    agri_mask = (
        np.isin(wc_data.astype(np.int32), agricultural_classes) & valid
        if wc_data is not None and wc_data.shape == s1_shape
        else None
    )
    mapped_agri_area = mask_area_km2(agri_mask, s1_transform) if agri_mask is not None else None
    exposure_mask = (
        (flood_mask & agri_mask) if flood_mask is not None and agri_mask is not None else None
    )
    exposed_area = mask_area_km2(exposure_mask, s1_transform) if exposure_mask is not None else None
    exposed_share = (
        exposed_area / mapped_agri_area * 100.0
        if exposed_area is not None and mapped_agri_area not in (None, 0.0)
        else None
    )

    repo.update_run(run, progress_percent=92, progress_message="Vectorizing map layers")
    from geoalchemy2.shape import from_shape

    from app.models import AnalysisLayer
    from app.services.layer_service import LAYER_DEFS

    layer_specs = [
        ("study_area", geom_wgs84, projected_area_km2(geom_wgs84, epsg), True),
        ("ml_flood", _mask_to_wgs84(flood_mask, s1_transform, utm_crs), detected_area, flood_mask is not None),
        ("baseline_flood", _mask_to_wgs84(baseline_mask, s1_transform, utm_crs), baseline_area, baseline_mask.any()),
        ("agricultural_exposure", _mask_to_wgs84(exposure_mask, s1_transform, utm_crs), exposed_area, exposure_mask is not None),
        ("permanent_water", _mask_to_wgs84(
            (jrc_data >= permanent_water_threshold) & np.isfinite(jrc_data) & aoi_mask
            if mask_applied else None,
            s1_transform, utm_crs), None, mask_applied),
    ]
    for layer_key, geom, layer_area, available in layer_specs:
        geometry_wkb = from_shape(simplify_if_large(geom), srid=4326) if geom is not None else None
        feature_count = 1 if geom is not None else 0
        repo.upsert_layer(
            AnalysisLayer(
                analysis_run_id=run.id,
                layer_key=layer_key,
                display_name=LAYER_DEFS[layer_key]["display_name"],
                available=bool(available and geom is not None and not geom.is_empty),
                geometry=geometry_wkb,
                storage_uri=str(run_dir) if available else None,
                feature_count=feature_count if available else None,
                area_km2=layer_area if available else None,
                metadata_json={},
            )
        )

    from app.models import AnalysisSummary

    repo.upsert_summary(
        AnalysisSummary(
            analysis_run_id=run.id,
            study_area_km2=round(projected_area_km2(geom_wgs84, epsg), 4),
            detected_flood_area_km2=round(detected_area, 4) if detected_area is not None else None,
            baseline_flood_area_km2=round(baseline_area, 4),
            mapped_agricultural_area_km2=round(mapped_agri_area, 4) if mapped_agri_area is not None else None,
            potentially_exposed_agricultural_area_km2=round(exposed_area, 4) if exposed_area is not None else None,
            potentially_exposed_agricultural_share_percent=round(exposed_share, 3) if exposed_share is not None else None,
            valid_pixel_count=valid_count,
            nodata_pixel_count=nodata_count,
            summary_metadata={
                "utm_epsg": epsg,
                "resolution_m": 10.0,
                "baseline_threshold": baseline_threshold,
                "permanent_water_threshold": permanent_water_threshold if mask_applied else None,
                "prepared_note": prepared_note,
                "observation_type": (observation or {}).get("type"),
                "observation_note": (observation or {}).get("note"),
                "not_peak_flood_extent": True,
                "permanent_water_rule": (
                    f"JRC occurrence >= {permanent_water_threshold}" if mask_applied else "not applied"
                ),
                "mapped_cropland_classes": agricultural_classes,
                "mapped_cropland_class_name": "ESA WorldCover class 40 (Cropland)",
                "exposure_terminology": "Mapped Cropland Potentially Exposed to Detected Water",
                "confirmed_crop_damage": False,
                "sar_unit": s1_unit,
                "sar_converted_to_db": s1_unit == "linear",
            },
        )
    )

    processing_config = {
        "sources": sources,
        "observation": observation,
        "sar_input": {
            "declared_unit": s1_unit,
            "converted_to_db": s1_unit == "linear",
            "conversion": "10*log10(clip(linear,1e-6,None))" if s1_unit == "linear" else "none",
            "preprocessed_raster": "s1_preprocessed.tif",
            "units_never_inferred": True,
        },
        "grid": {
            "crs": utm_crs,
            "transform": list(s1_transform)[:6],
            "width": int(s1_shape[1]),
            "height": int(s1_shape[0]),
            "resolution_m": 10.0,
            "nodata": "nan",
        },
        "permanent_water": {
            "applied": mask_applied,
            "source": "JRC Global Surface Water v1.4" if mask_applied else None,
            "threshold": permanent_water_threshold if mask_applied else None,
            "rule": "occurrence >= threshold treated as permanent water",
            "note": None if mask_applied else "Permanent-water mask was not available for this run.",
        },
        "baseline": {
            "method": "otsu_thresholding",
            "band": "VH_dB",
            "water_is_low": True,
            "threshold": baseline_threshold,
        },
        "ml": {
            "model_dir": model_dir,
            "model_version": _model_version(model_dir) if model_dir else None,
            "checkpoint_sha256": _checkpoint_hash(model_dir) if model_dir else None,
            "probability_threshold": probability_threshold,
            "note": "Custom U-Net checkpoint was not available; ML layer omitted." if not model_dir else None,
        },
        "agricultural_classes": agricultural_classes,
        "worldcover_version": "v200 (2021)",
        "agricultural_class_name": "ESA WorldCover class 40 (Cropland)",
        "utm_epsg": epsg,
        "prepared_bundle": prepared_note,
        "code_git_sha": _git_sha(),
    }

    has_ml = model_dir is not None
    repo.update_run(
        run,
        status="completed",
        completed_at=datetime.now(timezone.utc),
        progress_percent=100,
        progress_message=None,
        model_name="HaorWatch Flood Classifier" if has_ml else "HaorWatch Flood Classifier",
        model_version=_model_version(model_dir) if has_ml else None,
        baseline_method="otsu_thresholding",
        permanent_water_mask_applied=mask_applied,
        permanent_water_threshold=permanent_water_threshold if mask_applied else None,
        processing_config=processing_config,
    )

    from app.services.report_service import ReportService

    report_service = ReportService(db)
    report_service.get_report(run.id, "bn")
    print(f"Analysis run {run.id} completed.")


def _model_version(model_dir: str | None) -> str | None:
    if not model_dir:
        return None
    manifest = Path(model_dir) / "manifest.json"
    if manifest.exists():
        try:
            return json.loads(manifest.read_text(encoding="utf-8")).get("model_version")
        except json.JSONDecodeError:
            return None
    config = Path(model_dir) / "config.json"
    if config.exists():
        try:
            return json.loads(config.read_text(encoding="utf-8")).get("model_version")
        except json.JSONDecodeError:
            return None
    return None


def _checkpoint_hash(model_dir: str | None) -> str | None:
    if not model_dir:
        return None
    checkpoint = Path(model_dir) / "checkpoint.pt"
    if not checkpoint.exists():
        return None
    import hashlib

    digest = hashlib.sha256()
    with open(checkpoint, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str | None:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None
    return None


def _mask_to_wgs84(mask, transform, utm_crs):
    if mask is None or not mask.any():
        return None
    geom = mask_to_geometry(mask, transform)
    if geom is None:
        return None
    from app.geospatial.reprojection import reproject_geometry

    return ensure_multipolygon(reproject_geometry(geom, utm_crs, "EPSG:4326"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the end-to-end flood analysis pipeline.")
    parser.add_argument("--slug", default=None)
    parser.add_argument("--start", help="Observation window start, ISO-8601")
    parser.add_argument("--end", help="Observation window end, ISO-8601")
    parser.add_argument("--run-id", default=None, help="Resume an existing queued/failed run")
    parser.add_argument("--model-dir", default=None, help="U-Net bundle directory; omit to skip ML")
    parser.add_argument("--orbit", default=None, choices=[None, "ASCENDING", "DESCENDING"])
    parser.add_argument("--prepared-dir", default=None, help="Directory with prepared s1/jrc/worldcover rasters")
    args = parser.parse_args(argv)

    settings = get_settings()
    slug = args.slug or settings.default_study_area_slug
    if not args.prepared_dir and not (args.start and args.end):
        parser.error("--start and --end are required unless --prepared-dir is used")
    run_pipeline(slug, args.start, args.end, args.run_id, args.model_dir, args.orbit, args.prepared_dir)


if __name__ == "__main__":
    main()
