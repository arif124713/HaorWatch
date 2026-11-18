from pathlib import Path

root = Path(r"C:\Users\UNIQUE\OneDrive\Desktop\HaorWatch-web\frontend\src")
env_expr = "${import.meta.env.VITE_CARTO_API_KEY ?? ''}"
old_url = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
new_url = old_url + "?api_key=" + env_expr

for rel in ["components/map/MiniOverviewMap.tsx", "pages/MapExplorerPage.tsx"]:
    f = root / rel
    text = f.read_text(encoding="utf-8")
    if old_url in text and "api_key=" not in text:
        f.write_text(text.replace(old_url, new_url), encoding="utf-8")
        print(rel, "patched")
    else:
        print(rel, "skipped (already patched or URL differs)")

vite_env = root / "vite-env.d.ts"
text = vite_env.read_text(encoding="utf-8")
if "VITE_CARTO_API_KEY" not in text:
    text = text.replace(
        "readonly VITE_STUDY_AREA_SLUG?: string;",
        "readonly VITE_STUDY_AREA_SLUG?: string;\nreadonly VITE_CARTO_API_KEY?: string;",
    )
    vite_env.write_text(text, encoding="utf-8")
    print("vite-env.d.ts updated")

env_file = Path(r"C:\Users\UNIQUE\OneDrive\Desktop\HaorWatch-web\frontend\.env")
env_text = env_file.read_text(encoding="utf-8")
if "VITE_CARTO_API_KEY" not in env_text:
    env_file.write_text(env_text.rstrip("\n") + "\nVITE_CARTO_API_KEY=cb1_5bwg_1_b481cad610d30884b63acf54\n", encoding="utf-8")
    print(".env updated")
