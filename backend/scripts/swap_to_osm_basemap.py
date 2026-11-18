from pathlib import Path

root = Path(r"C:\Users\UNIQUE\OneDrive\Desktop\HaorWatch-web\frontend\src")
carto_templated = "url={`https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png?api_key=${import.meta.env.VITE_CARTO_API_KEY ?? \"\"}`}"
osm_templated = "url={`https://tile.openstreetmap.org/{z}/{x}/{y}.png`}"

for rel in ["components/map/MiniOverviewMap.tsx", "pages/MapExplorerPage.tsx"]:
    f = root / rel
    text = f.read_text(encoding="utf-8")
    if carto_templated in text:
        f.write_text(text.replace(carto_templated, osm_templated), encoding="utf-8")
        print(rel, "-> OSM keyless basemap (Carto key kept in .env for later)")
    elif osm_templated in text:
        print(rel, "already OSM")
    else:
        print(rel, "URL NOT FOUND — inspect")
