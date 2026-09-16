# Final project presentation

**[Open the slides](slides.pdf)** — 27 slides, Samuel Barber, 25 June 2026.

The presentation covers NIR-to-RGB translation, the capture rig, training and
distillation, downstream evaluation, and deployment on Raspberry Pi 5 / Hailo-8L.
The [live Pi demo](live-pi-demo.mp4) is included alongside the slides.

## Build

Run from this directory with XeLaTeX and a current Beamer installation:

```bash
xelatex -jobname=slides presentation.tex
xelatex -jobname=slides presentation.tex
```

The source uses the bundled Imperial theme, `Fonts/`, `Images/`, and shared
figures in `../report/imgs/`. Keep that relative layout when rebuilding.

The Imperial theme, logos and fonts retain their [license](LICENSE.txt) and
[font license](Fonts/LICENSE.txt).
