"""Wrap the studio page source into a standalone frontend/index.html.

The page source (frontend/studio.html) is written as a body fragment so the same
file can be published as a claude.ai Artifact; this script adds the document shell
for self-hosting.

    python scripts/build_frontend.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src = (ROOT / "frontend" / "studio.html").read_text(encoding="utf-8")
shell = (
    "<!doctype html>\n<html lang=\"zh-CN\">\n<head>\n<meta charset=\"utf-8\">\n"
    "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1,viewport-fit=cover\">\n"
    "<style>body{margin:0}[hidden]{display:none!important}</style>\n</head>\n<body>\n"
)
(ROOT / "frontend" / "index.html").write_text(shell + src + "\n</body>\n</html>\n", encoding="utf-8")
print("wrote frontend/index.html")
