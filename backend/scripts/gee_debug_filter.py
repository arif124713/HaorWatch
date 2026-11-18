import json

import ee
from app.core.config import get_settings

settings = get_settings()
ee.Initialize(project=settings.google_earth_engine_project)

aoi_geojson = json.load(open(r"data\hakaluki_aoi.geojson", encoding="utf-8"))["features"][0]["geometry"]
geom = ee.Geometry(aoi_geojson)
print("geometry valid, type:", geom.type().getInfo())

col = ee.ImageCollection("COPERNICUS/S1_GRD")
steps = {
    "filterBounds": col.filterBounds(geom),
}
steps["filterBounds+date"] = steps["filterBounds"].filterDate("2026-06-01", "2026-10-31")
steps["+IW"] = steps["filterBounds+date"].filter(ee.Filter.eq("instrument_mode", "IW"))
steps["+VV"] = steps["+IW"].filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
steps["+VH"] = steps["+VV"].filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
for name, collection in steps.items():
    print(name, "->", collection.size().getInfo())

sample = steps["filterBounds+date"].first().getInfo()
props = sample.get("properties", {})
print("sample instrument_mode:", props.get("instrument_mode"))
print("sample polarisation:", props.get("transmitterReceiverPolarisation"))
print("sample orbit pass:", props.get("orbitProperties_pass"))
