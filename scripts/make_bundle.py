"""Package code + frozen examples + embedding cache as the Kaggle dataset used by kaggle/ollama_runs.
Uploading individual files (not folders) keeps Kaggle from re-zipping or renaming anything."""
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "kaggle" / "ollama_runs" / "bundle"
out.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(out / "sketchrag_bundle.zip.bin", "w", zipfile.ZIP_DEFLATED) as z:
    for pattern in ("src/**/*.py", "scripts/*.py", "data_raw/examples/*.jsonl", "pyproject.toml"):
        for p in ROOT.glob(pattern):
            if "__pycache__" not in p.parts:
                z.write(p, p.relative_to(ROOT).as_posix())
shutil.copy(ROOT / "cache" / "emb.sqlite", out / "emb.sqlite")
(out / "dataset-metadata.json").write_text(
    '{"title": "sketchrag-bundle", "id": "rranjithkumariiitk/sketchrag-bundle", "licenses": [{"name": "other"}]}')
for p in sorted(out.iterdir()):
    print(f"{p.name:30s} {p.stat().st_size / 2**20:8.1f} MB")
