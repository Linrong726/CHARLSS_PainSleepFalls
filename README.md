# CHARLS pain, restless sleep and subsequent falls

Analysis and figure code for the study of 2015 joint pain and restless-sleep status among older China Health and Retirement Longitudinal Study participants who reported both symptoms in 2011, with falls subsequently reported in the 2018 interview. The release uses the frozen Stage 1 v1.1 aggregate outputs and Stage 2 publication display code. The estimands are reported associations in the defined samples.

## Reproduce the figures from aggregates

In an isolated Python 3.12 environment, install `requirements-figures.txt` and make the Poppler tools `pdffonts` and `pdftotext` available on PATH. Run these commands from the repository directory containing this README. Use a fresh figure output directory.

```sh
python -m pip install -r requirements-figures.txt
mkdir -p build/qa
python scripts/check_inputs.py --aggregate
python figures/figures_stage2.py data/aggregate build/figures
python figures/audit_figures_stage2.py data/aggregate build/figures build/qa/FIGURE_QA.json
```

The figures and the generated figure-source map are written to `build/figures`. The closeout ran these commands and verified all seven figures and 117 source rows without fitting models. See the [verification record](validation/FIGURE_QA.json) and [reproduction details](docs/REPRODUCE.md). The [aggregate field guide](docs/AGGREGATES.md) explains the result scales and relevant denominators. The final, formatted XLSX is a separate manuscript companion, Additional file 3.

## Rebuild the analysis with authorized data

The full analysis route requires authorized local access to the exact CHARLS source releases and the Version D codebook. See the [input inventory](docs/INPUTS.md) and [reproduction details](docs/REPRODUCE.md) for setup, output locations and interpretation. The closeout verified the Python source, imports and help entry points. Historical model runs are documented separately from this release check.

This package contains author analysis code, documentation and aggregate outputs. Data providers distribute the participant-level inputs and their official materials under their own terms. Observe any applicable repository license and data-provider terms.
