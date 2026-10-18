# SketchRAG final report: LaTeX source

IEEE two-column conference format (`IEEEtran`), 7 pages.

| File | What it is |
|---|---|
| `sketchrag.tex` | The paper |
| `refs.bib` | Bibliography (BibTeX, IEEEtran style) |
| `numbers.tex` | Every number quoted in the text, as LaTeX macros (generated) |
| `tables/*.tex` | All tables (generated) |
| `figs/*.pdf`, `figs/demo_query.png` | Figures (generated; the PNG is a screenshot of the demo console) |
| `make_assets.py` | Regenerates `numbers.tex`, `tables/` and `figs/` from the project's `results/` folder |

Build:

```
latexmk -pdf sketchrag.tex          # or: pdflatex, bibtex, pdflatex, pdflatex
```

To regenerate tables, figures and numbers after changing results (run from the project root):

```
python report/paper/make_assets.py
```

Nothing in the tables or quoted numbers is typed by hand; edit the prose in `sketchrag.tex` freely,
but change numbers by re-running `make_assets.py`.
