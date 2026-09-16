# Final report

[Read the final thesis](main.pdf): **Edge-Deployable NIR-to-RGB Translation as a
Pre-Processor for Pre-Trained Vision Models**, Samuel Barber, Imperial College
London, 2026.

`main.pdf` is the submitted report. `main.tex`, `chapters/`, `references.bib`,
`ic_eee_thesis.cls`, the Imperial logo and `imgs/` are its build inputs. Figures
are also shared by the presentation and paper. `figtools/` contains reusable
plotting utilities; their docstrings describe the required experiment inputs.

## Build

Use a TeX distribution with LuaLaTeX, Biber and the packages requested by the
source. Run from this directory:

```bash
lualatex main.tex
biber main
lualatex main.tex
lualatex main.tex
```

The bibliography is required to rebuild the report and paper. It is retained
with the source. Compiled intermediates are ignored by Git.
