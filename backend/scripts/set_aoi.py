import argparse
import json
from pathlib import Path

from geoalchemy2.shape import from_shape
from shapely.geometry import shape
from shapely.ops import unary_union

from app.core.database import get_session_factory
from app.geospatial.area import projected_area_km2
from app.geospatial.vectorize import ensure_multipolygon
from app.models import StudyArea


def load_geometry(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") == "FeatureCollection":
        geoms = [shape(feature["geometry"]) for feature in data.get("features", [])]
    elif data.get("type") == "Feature":
        geoms = [shape(data["geometry"])]
    else:
        geoms = [shape(data)]
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        raise SystemExit("No usable geometry found in the file.")
    merged = unary_union(geoms) if len(geoms) > 1 else geoms[0]
    multipolygon = ensure_multipolygon(merged)
    if multipolygon is None:
        raise SystemExit("Geometry could not be interpreted as a polygon.")
    return multipolygon


def main():
    parser = argparse.ArgumentParser(description="Set the AOI boundary of a study area from a GeoJSON file.")
    parser.add_argument("--slug", default=None, help="Study area slug (defaults to DEFAULT_STUDY_AREA_SLUG)")
    parser.add_argument("--file", required=True, help="Path to GeoJSON (Feature, FeatureCollection, or raw geometry)")
    parser.add_argument("--source-note", default="AOI geometry supplied by operator", help="Provenance note stored in metadata")
    args = parser.parse_args()

    from app.core.config import get_settings

    settings = get_settings()
    slug = args.slug or settings.default_study_area_slug
    geometry = load_geometry(Path(args.file))
    area_km2 = projected_area_km2(geometry)

    factory = get_session_factory()
    with factory() as db:
        area = db.query(StudyArea).filter(StudyArea.slug == slug).first()
        if area is None:
            raise SystemExit(f"Study area '{slug}' not found. Run scripts.seed_demo first.")
        area.boundary = from_shape(geometry, srid=4326)
        area.area_km2 = round(area_km2, 3)
        metadata = dict(area.metadata_json or {})
        metadata.update({
            "aoi_status": "confirmed_from_operator_geometry",
            "aoi_source_file": Path(args.file).name,
            "aoi_source_note": args.source_note,
            "aoi_parts": len(geometry.geoms),
        })
        area.metadata_json = metadata
        db.commit()
        print(f"AOI set for '{slug}': {area_km2:.2f} km2, {len(geometry.geoms)} polygon part(s).")


if __name__ == "__main__":
    main()
