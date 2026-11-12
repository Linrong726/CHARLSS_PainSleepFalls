# Reproduce figures and reconstruct the analysis

## Reproduce the publication figures

Obtain the manuscript's **Additional file 4**, `CHARLS_PainSleepFalls_Reproducibility_Inputs_v1_0.zip`, from the manuscript attachment or published supplementary materials. Extract the archive and follow its README. From the extracted input-package directory, run `python verify_install.py` and then `python verify_install.py --repository "/absolute/path/to/CHARLSS_PainSleepFalls" --install`, replacing the path with your code checkout. The utility verifies the nine frozen input hashes and compatible scientific code, copies missing files into the required relative paths and refuses to overwrite different existing files. It does not obtain participant-level data or run models.

The six figure CSVs will be under `data/aggregate` and the frozen source map at `reference/FIGURE_SOURCE_LONG.csv`. From this repository root, use Python 3.12 in an isolated environment. Install Poppler and make its `pdffonts` command available on `PATH`. Use the included Matplotlib default DejaVu Sans font. Poppler's `pdftoppm` or `pdftocairo` can also render vector output for visual inspection.

```sh
python -m pip install -r requirements-figures.txt
python -c "from pathlib import Path; Path('build/qa').mkdir(parents=True, exist_ok=True)"
python figures/figures_stage2.py data/aggregate build/figures
python figures/audit_figures_stage2.py data/aggregate build/figures build/qa/FIGURE_QA.json
python -c "from pathlib import Path; assert Path('build/figures/FIGURE_SOURCE_LONG.csv').read_bytes() == Path('reference/FIGURE_SOURCE_LONG.csv').read_bytes()"
```

The figure generator reads the frozen `results_master.csv`, `flow.csv`, `followup.csv`, `missing.csv`, `selection.csv` and `chronic.csv`. It creates seven PDFs, seven PNGs and figure-source mappings. The audit checks the aggregate provenance, plotted numerical values and annotations, figure size and resolution, and font embedding. Additional file 4 provides `INPUT_PACKAGE_VERIFICATION.json`, which records the observed installation and figure-reproduction checks and software versions; its participant-level models were not rerun. The 117 figure-source rows are the comparison target. Figure binaries can vary with the rendering environment.

The `se` field is the cluster-robust standard error on the log(RR) coefficient scale for RR rows. For PROB and RD rows it is the delta-method standard error on the original probability or probability-difference scale. Probability confidence intervals use a logit transformation. The companion submission's Additional file 3 is the formatted aggregate workbook. The figure commands use the CSV data directly.

## Reconstruct the analysis

Obtain the authorized data and codebook specified in [Input files and versions](INPUTS.md). Install the two exact frozen verification records supplied in Additional file 4 at the relative paths listed there; the archive's `verify_install.py` utility installs both records together with the figure inputs. Use an isolated Python 3.12 environment and install the historical package versions from [requirements-analysis.txt](../requirements-analysis.txt). The original Windows run used Python 3.12.14. The scientific entry point retains the frozen cohort, release and design checks. Because it validates design digests, a mismatch should be resolved by checking data release and software environment identities before comparing results.

Copy [config.example.json](../analysis/code/config.example.json) to a local `config.local.json` outside this repository. Set `data_root` to the absolute directory containing your authorized inputs. Choose two new output paths outside this repository with existing parent directories. Keep the two destinations separate, with neither contained in the other. In the example, replace the paths with your own.

```sh
python -m pip install -r requirements-analysis.txt
python analysis/code/check_inputs.py --config /path/to/config.local.json --out /path/to/new-analysis-output --private /path/to/new-restricted-output
python analysis/code/stage1.py --config /path/to/config.local.json --out /path/to/new-analysis-output --private /path/to/new-restricted-output
```

The preflight checks input and codebook file hashes, destination paths and importability of the main runtime. The analysis rebuilds the cohort, runs the frozen specification and writes the contrasts and absolute estimates to `results/results_master.csv`. It also produces verification evidence and Stage 1 diagnostic figures. The `--private` destination contains a local membership audit with participant keys and must remain in restricted storage. Review every newly created output before sharing it.

The analysis uses modified Poisson regression without an offset or survey weights, with a 2011 community-clustered CR1 sandwich. Nine fitted models are distinguished from the `ABS_POISSON` standardization. The optional Stage 1 `--node` workbook export requires a compatible Node installation and `@oai/artifact-tool` located via `STAGE1_NODE_MODULES`. The standard commands omit it. The Stage 2 workbook and formatted manuscript are separate submission files.

Historical Stage 1 v1.1 evidence documents comparison of the original and replayed analysis runs from 10 November 2026. For the current closeout, the figure route was run against the separate frozen aggregate inputs and the analysis entry point, local imports and dependencies were checked. Participant-level analysis was not executed for this distribution.
