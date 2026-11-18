import re
from pathlib import Path

text = Path(r"C:\Users\UNIQUE\OneDrive\Desktop\HaorWatch-web\frontend\src\lib\api-client.ts").read_text(encoding="utf-8")
paths = sorted(set(re.findall(r"[`'\"](/[^`'\"]+)[`'\"]", text)))
for p in paths:
    print(p)
