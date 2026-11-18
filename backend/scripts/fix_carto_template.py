from pathlib import Path

root = Path(r"C:\Users\UNIQUE\OneDrive\Desktop\HaorWatch-web\frontend\src")
broken = 'url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png?api_key=${import.meta.env.VITE_CARTO_API_KEY ?? \'{}\'}"'.replace("{}", "")
fixed = "url={`https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png?api_key=${import.meta.env.VITE_CARTO_API_KEY ?? \"\"}`}"

for rel in ["components/map/MiniOverviewMap.tsx", "pages/MapExplorerPage.tsx"]:
    f = root / rel
    text = f.read_text(encoding="utf-8")
    if broken in text:
        f.write_text(text.replace(broken, fixed), encoding="utf-8")
        print(rel, "fixed to template literal")
    elif fixed in text:
        print(rel, "already correct")
    else:
        print(rel, "PATTERN NOT FOUND — inspect manually")
