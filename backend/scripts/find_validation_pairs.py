import ee

from app.core.config import get_settings

settings = get_settings()
ee.Initialize(project=settings.google_earth_engine_project)

aoi = ee.Geometry(json_aoi) if False else ee.Geometry({
    "type": "Polygon",
    "coordinates": [[[91.95, 24.58], [92.13, 24.58], [92.13, 24.74], [91.95, 24.74], [91.95, 24.58]]],
})

s1 = (
    ee.ImageCollection("COPERNICUS/S1_GRD")
    .filterBounds(aoi)
    .filterDate("2026-07-01", "2026-09-30")
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
)
dates = s1.aggregate_array("system:time_start").getInfo()
import datetime

s1_days = sorted({datetime.datetime.fromtimestamp(d / 1000, datetime.UTC).date().isoformat() for d in dates})
print("S1 acquisition days:", s1_days)

s2 = (
    ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
    .filterBounds(aoi)
    .filterDate("2026-07-01", "2026-09-30")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 12))
)
s2_info = s2.aggregate_array("system:time_start").getInfo()
s2_days = sorted({datetime.datetime.fromtimestamp(d / 1000, datetime.UTC).date().isoformat() for d in s2_info})
print("S2 low-cloud days:", s2_days)

from datetime import date

pairs = []
for s1_day in s1_days:
    s1_d = date.fromisoformat(s1_day)
    candidates = [d for d in s2_days if abs((date.fromisoformat(d) - s1_d).days) <= 3]
    if candidates:
        pairs.append((s1_day, candidates))
print("S1/S2 pairs within 3 days:", pairs)
