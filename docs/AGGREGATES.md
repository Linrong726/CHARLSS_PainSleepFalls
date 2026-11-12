# Aggregate results and figure sources

The manuscript's **Additional file 3** is a formatted aggregate-results workbook. **Additional file 4**, `CHARLS_PainSleepFalls_Reproducibility_Inputs_v1_0.zip`, supplies the nine exact frozen inputs needed by the documented workflows: six figure CSVs, the frozen figure-source map and two aggregate analysis-validation records. Its README, `INPUTS_MANIFEST.csv` and verification utility give their paths and SHA-256 hashes.

The six CSVs are byte-identical to the corresponding frozen Stage 1 v1.1 results. This minimal input archive is different from the historical author archive containing all 35 frozen result CSVs; it does not claim to distribute all 35 CSVs. The remaining reported aggregate results and field definitions are provided in Additional file 3. The documented figure commands do not read the workbook or require conversion of its worksheets to CSV.

| Files in Additional file 4 | Contents |
| --- | --- |
| `data/aggregate/results_master.csv` | Estimates, confidence intervals, contrasts, and model and sample labels used by the figures |
| `data/aggregate/flow.csv`, `followup.csv`, `missing.csv`, `selection.csv`, `chronic.csv` | Aggregate cohort composition, follow-up, missingness, selection and condition-source comparisons used by the figures |
| `reference/FIGURE_SOURCE_LONG.csv` | Frozen source-row keys and displayed values for all seven figures |
| `analysis/code/reference/v121/01_patch/analysis_invariance.json`, `analysis/code/reference/v21/08_QA_AND_REPRODUCIBILITY/nonfit_summary.json` | Aggregate frozen verification records required by the analysis entry point |

In `results_master.csv`, RR is a risk ratio for reporting a fall. For RR rows, `se` is the cluster-robust standard error on the log(RR) coefficient scale. For PROB and RD rows, it is the delta-method standard error on the original probability or probability-difference scale. Probability confidence intervals use a logit transformation. The absolute probabilities and differences are proportions in the CSV. The figures multiply them by 100 and show percentages or percentage points, respectively.

Additional file 4's `reference/FIGURE_SOURCE_LONG.csv` supplies table and row keys for the displayed numbers. Its 117 rows cover four main and three supplementary figures.
