# Condensed paper draft

[Read the WACV-style draft](wacv2027_draft.pdf). The
[final report](../report/main.pdf) is the authoritative thesis.

Build from this directory with LuaLaTeX and Biber:

```bash
lualatex wacv2027_draft.tex
biber wacv2027_draft
lualatex wacv2027_draft.tex
lualatex wacv2027_draft.tex
```

The source shares `../report/references.bib` and `../report/imgs/`.
