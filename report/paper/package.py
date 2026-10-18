"""Bundle the submission: report/SketchRAG_Final_Report.pdf and report/SketchRAG_Final_Report_source.zip."""
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
files = [HERE / f for f in ("sketchrag.tex", "refs.bib", "numbers.tex", "make_assets.py", "README.md")]
files += sorted((HERE / "tables").glob("*.tex")) + sorted(p for p in (HERE / "figs").glob("*") if p.suffix in (".pdf", ".png"))
out = HERE.parent / "SketchRAG_Final_Report_source.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, "SketchRAG_Final_Report_source/" + f.relative_to(HERE).as_posix())
shutil.copy(HERE / "sketchrag.pdf", HERE.parent / "SketchRAG_Final_Report.pdf")
print(len(files), "files ->", out)
