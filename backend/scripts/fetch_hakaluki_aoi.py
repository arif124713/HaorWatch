import httpx
import json
from pathlib import Path

base = "https://www.arcgisbd.com/server/rest/services/ADB005/CRIIPS/MapServer/147/query"
r = httpx.get(
    base,
    params={
        "where": "UPPER(HAOR_NAME) LIKE '%HAKALUKI%'",
        "outFields": "HAOR_NAME,AREA_SQKM,TYPE",
        "returnGeometry": "true",
        "outSR": 4326,
        "f": "json",
    },
    timeout=60,
)
data = r.json()
feats = data.get("features", [])
print("matches:", len(feats))
for f in feats:
    print("  attrs:", f.get("attributes"))
    rings = f.get("geometry", {}).get("rings", [])
    print("  rings:", len(rings), "ring0 pts:", len(rings[0]) if rings else 0)

features = []
for f in feats:
    rings = f["geometry"]["rings"]
    outer = [ring for ring in rings if len(ring) >= 4]
    if not outer:
        continue
    polygons = [[ring] for ring in outer]
    features.append({
        "type": "Feature",
        "properties": f.get("attributes", {}),
        "geometry": {"type": "MultiPolygon", "coordinates": polygons},
    })

fc = {"type": "FeatureCollection", "features": features}
dest = Path(r"data\hakaluki_aoi.geojson")
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(fc, indent=1), encoding="utf-8")
print("saved GeoJSON with", len(features), "feature(s) ->", dest)
