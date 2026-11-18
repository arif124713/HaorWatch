import ee
from app.core.config import get_settings

settings = get_settings()
ee.Initialize(project=settings.google_earth_engine_project)

aoi = ee.Geometry.Rectangle([91.9, 24.35, 92.35, 24.85])
col = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi)

for start, end in [
    ("2024-01-01", "2024-12-31"),
    ("2025-01-01", "2025-06-30"),
    ("2025-07-01", "2025-12-31"),
    ("2026-01-01", "2026-05-31"),
    ("2026-06-01", "2026-10-31"),
    ("2026-11-01", "2026-11-18"),
]:
    count = col.filterDate(start, end).size().getInfo()
    print(f"{start} .. {end}: {count}")

latest = col.limit(1).aggregate_max("system:time_start").getInfo()
import datetime
print("latest acquisition over AOI:", datetime.datetime.utcfromtimestamp(latest / 1000).isoformat())

jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
print("JRC OK:", jrc.getInfo()["id"])
wc = ee.Image("ESA/WorldCover/v200")
print("WorldCover OK:", wc.getInfo()["id"])
