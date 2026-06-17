from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "presentation_assets"
DEST_ROOT = ROOT / "docs"

COPY_DIRS = [
    "therapy_feedback_showcase",
    "jefferson_demo",
    "squat_demo",
]
def replace_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def write_root_index() -> None:
    target = DEST_ROOT / "index.html"
    target.write_text(
        """<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ARKIST Motion AI Demo</title>
  <meta http-equiv="refresh" content="0; url=therapy_feedback_showcase/index.html">
  <style>
    body {
      margin: 0;
      padding: 48px 24px;
      font-family: Arial, Helvetica, sans-serif;
      background: #f6f4ef;
      color: #16202a;
    }
    main {
      max-width: 760px;
      margin: 0 auto;
      background: #fff;
      border: 1px solid #d9d4c9;
      border-radius: 8px;
      padding: 28px;
    }
    h1 {
      margin: 0 0 12px;
      font-size: 30px;
    }
    p {
      margin: 0;
      font-size: 17px;
      line-height: 1.5;
    }
    a {
      color: #1d4ed8;
    }
  </style>
</head>
<body>
  <main>
    <h1>ARKIST Motion AI Demo</h1>
    <p>Weiterleitung zur Praesentationsseite. Falls die Weiterleitung nicht automatisch erfolgt, bitte <a href="therapy_feedback_showcase/index.html">hier klicken</a>.</p>
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    for dirname in COPY_DIRS:
        src = SOURCE_ROOT / dirname
        dst = DEST_ROOT / dirname
        if not src.exists():
            raise FileNotFoundError(f"Missing source directory: {src}")
        replace_dir(dst)
        shutil.copytree(src, dst)

    write_root_index()
    (DEST_ROOT / ".nojekyll").write_text("", encoding="utf-8")
    print(f"saved_pages_site: {DEST_ROOT}")


if __name__ == "__main__":
    main()
