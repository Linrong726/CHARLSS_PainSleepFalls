# Aggregate results and figure sources

The separate reproducibility package contains 35 `data/aggregate` CSVs byte-identical to the frozen Stage 1 v1.1 results and a verification record with their SHA-256 hashes. The six inputs for the figure route are listed in [Reproduce figures and reconstruct the analysis](REPRODUCE.md).

| Files | Contents |
| --- | --- |
| `results_master.csv`, `sample_groups.csv` | Estimates, confidence intervals, contrasts, model and sample labels, and group counts |
| `baseline.csv`, `flow.csv`, `followup*.csv`, `missing*.csv`, `month_*.csv`, `s3*.csv`, `selection*.csv`, `chronic.csv`, `walk_source.csv`, `cohort_linkage_qc.csv` | Aggregate cohort composition, follow-up, missingness, selection, and source checks |
| `model_*.csv`, `probability_*.csv` | Model specifications, estimates, covariance, diagnostics and scenario summaries |
| `numeric_index.csv`, `table_*.csv` | Source-row links and formatted table values |

In `results_master.csv`, RR is a risk ratio for reporting a fall. For RR rows, `se` is the cluster-robust standard error on the log(RR) coefficient scale. For PROB and RD rows, it is the delta-method standard error on the original probability or probability-difference scale. Probability confidence intervals use a logit transformation. The absolute probabilities and differences are proportions in the CSV. The figures multiply them by 100 and show percentages or percentage points, respectively.

The separate package’s `reference/FIGURE_SOURCE_LONG.csv` supplies table and row keys for the displayed numbers. Its 117 rows cover four main and three supplementary figures.
