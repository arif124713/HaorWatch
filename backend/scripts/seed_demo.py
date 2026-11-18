import argparse
from pathlib import Path

from app.core.config import get_settings
from app.core.database import get_session_factory
from app.models import DatasetAsset, StudyArea

SOURCE_CATALOGUE = [
    {
        "dataset_key": "sentinel1_sar",
        "display_name": "Sentinel-1 SAR (GRD, IW, VV+VH)",
        "provider": "Copernicus / ESA",
        "data_type": "raster",
        "source_url": "https://dataspace.copernicus.eu/",
        "license_note": "Copernicus Open Data Licence; free and open access.",
        "resolution_m": 10.0,
    },
    {
        "dataset_key": "sen1floods11",
        "display_name": "Sen1Floods11 labelled flood dataset",
        "provider": "Cloud to Street / NASA",
        "data_type": "raster",
        "source_url": "https://github.com/cloudtostreet/Sen1Floods11",
        "license_note": "Check repository LICENSE before training and record exact terms.",
        "resolution_m": 10.0,
    },
    {
        "dataset_key": "jrc_gsw",
        "display_name": "JRC Global Surface Water",
        "provider": "European Commission Joint Research Centre",
        "data_type": "raster",
        "source_url": "https://global-surface-water.appspot.com/",
        "source_version": "1.4",
        "license_note": "CC BY 4.0.",
        "resolution_m": 30.0,
    },
    {
        "dataset_key": "esa_worldcover",
        "display_name": "ESA WorldCover",
        "provider": "European Space Agency",
        "data_type": "raster",
        "source_url": "https://esa-worldcover.org/",
        "source_version": "v200 (2021)",
        "license_note": "CC BY 4.0.",
        "resolution_m": 10.0,
    },
]


def seed(session_factory=None) -> None:
    settings = get_settings()
    factory = session_factory or get_session_factory()
    with factory() as db:
        slug = settings.default_study_area_slug
        existing = db.query(StudyArea).filter(StudyArea.slug == slug).first()
        if existing is None:
            area = StudyArea(
                slug=slug,
                name="Hakaluki Haor",
                division="Sylhet",
                districts=["Moulvibazar", "Sylhet"],
                country_code="BD",
                boundary=None,
                area_km2=None,
                is_active=True,
                metadata_json={
                    "aoi_status": "pending_confirmation",
                    "note": "Confirm the exact pilot AOI geometry before producing spatial results.",
                    "approx_centre_lonlat": [92.0, 24.6],
                },
            )
            db.add(area)
            print(f"Seeded study area '{slug}' with AOI pending confirmation.")
        else:
            area = existing
            print(f"Study area '{slug}' already present.")

        for item in SOURCE_CATALOGUE:
            exists = (
                db.query(DatasetAsset)
                .filter(DatasetAsset.dataset_key == item["dataset_key"], DatasetAsset.study_area_id == area.id)
                .first()
            )
            if exists is not None:
                continue
            db.add(DatasetAsset(study_area_id=area.id, **item))
        db.commit()
        print("Source catalogue seeded.")


def seed_demo_area(session_factory=None) -> None:
    import json
    from pathlib import Path

    import rasterio
    from geoalchemy2.shape import from_shape
    from shapely.geometry import Polygon

    from app.geospatial.area import projected_area_km2

    s1_path = Path("data/prepared/ghana-demo/s1.tif")
    manifest_path = Path("data/prepared/ghana-demo/manifest.json")
    if not s1_path.exists():
        raise SystemExit("Demo bundle missing: data/prepared/ghana-demo/s1.tif")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    with rasterio.open(s1_path) as src:
        west, south, east, north = src.bounds
    footprint = Polygon([(west, south), (east, south), (east, north), (west, north)])
    factory = session_factory or get_session_factory()
    with factory() as db:
        existing = db.query(StudyArea).filter(StudyArea.slug == "sen1floods11-ghana-demo").first()
        if existing is not None:
            print("Demo study area already present.")
            return
        area = StudyArea(
            slug="sen1floods11-ghana-demo",
            name="Sen1Floods11 Ghana Demo (prepared)",
            division="Demo",
            districts=[],
            country_code="GH",
            boundary=from_shape(__import__("shapely.geometry", fromlist=["MultiPolygon"]).MultiPolygon([footprint]), srid=4326),
            area_km2=round(projected_area_km2(footprint), 3),
            is_active=True,
            metadata_json={
                "aoi_status": "confirmed_from_source_chip",
                "prepared": True,
                "source_chip": manifest.get("source_chip"),
                "split": manifest.get("split"),
                "note": "Prepared demonstration analysis on a real Sen1Floods11 test chip; not the pilot haor AOI.",
            },
        )
        db.add(area)
        db.commit()
        print("Seeded demo study area 'sen1floods11-ghana-demo' with real chip footprint.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the pilot study area and source catalogue.")
    parser.add_argument("--include-demo", action="store_true", help="Also seed the prepared Sen1Floods11 demo study area")
    args = parser.parse_args()
    seed()
    if args.include_demo:
        seed_demo_area()
