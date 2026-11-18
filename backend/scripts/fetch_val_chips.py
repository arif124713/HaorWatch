import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

csv = httpx.get(
    "https://storage.googleapis.com/sen1floods11/v1.1/splits/flood_handlabeled/flood_valid_data.csv",
    timeout=30,
).text
rows = [line.split(",")[0].strip() for line in csv.splitlines() if line.strip()]
ghana = [r for r in rows if r.startswith("Ghana")][:10]
india = [r for r in rows if r.startswith("India")][:10]
base = "https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/"
out = Path(r"data\prepared\val-chips")
out.mkdir(parents=True, exist_ok=True)
for fname in ghana + india:
    for layer in ["S1Hand", "LabelHand"]:
        target = fname.replace("S1Hand", layer)
        p = out / target
        if not p.exists():
            r = httpx.get(base + f"{layer}/{target}", timeout=60, follow_redirects=True)
            r.raise_for_status()
            p.write_bytes(r.content)
print("ready:", len(list(out.glob("*S1Hand.tif"))), "val chips")
