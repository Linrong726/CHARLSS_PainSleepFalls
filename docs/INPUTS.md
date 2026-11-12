# Input files and versions

Apply through the [official CHARLS data access website](https://charls.pku.edu.cn/en/) for the original 2011, 2015 and 2018 waves. Obtain Harmonized CHARLS Version D through [CHARLS Harmonized Data](https://charls.charlsdata.com/pages/data/harmonized_charls/en.html) or the [Gateway to Global Aging Data](https://g2aging.org/), following each provider's access and use terms. The pipeline requires the byte identities in [input_lock.json](../analysis/code/reference/input_lock.json). Use the named release files and preserve the relative paths and letter case.

| Source | Relative path under `data_root` |
| --- | --- |
| Harmonized D linkage and covariates | `Harmonized CHARLS/H_CHARLS_D_Data.zip` |
| 2011 health_status_and_functioning | `2011 CHARLS Wave1/household_and_community_questionnaire_data/health_status_and_functioning.dta` |
| 2011 demographic_background | `2011 CHARLS Wave1/household_and_community_questionnaire_data/demographic_background.dta` |
| 2011 weight | `2011 CHARLS Wave1/household_and_community_questionnaire_data/weight.dta` |
| 2015 health_status_and_functioning | `2015 CHARLS Wave3/CHARLS2015r/Health_Status_and_Functioning.dta` |
| 2015 demographic_background | `2015 CHARLS Wave3/CHARLS2015r/Demographic_Background.dta` |
| 2015 sample_infor | `2015 CHARLS Wave3/CHARLS2015r/Sample_Infor.dta` |
| 2018 health_status_and_functioning | `2018 CHARLS Wave4/CHARLS2018r/Health_Status_and_Functioning.dta` |
| 2018 Cognition | `2018 CHARLS Wave4/CHARLS2018r/Cognition.dta` |
| 2018 demographic_background | `2018 CHARLS Wave4/CHARLS2018r/Demographic_Background.dta` |
| 2018 sample_infor | `2018 CHARLS Wave4/CHARLS2018r/Sample_Infor.dta` |

The weight file is a locked input even though the fitted models do not use survey weights.

Place the Harmonized Version D codebook (June 2021, regular waves 2011–2018) at `Harmonized CHARLS/Harmonized_CHARLS_D.pdf`. The expected SHA-256 is `40957f6e14b0f60aac7570aa59ed7a0aa27a1870cbc8b90b3b48207c74603a51`. The historical official listing's SHA-1 was `af828e94462f26f56cb0b2c188a9c3dad5cfcac0`. The inherited tracking module records this documentation in source provenance.

The separate historical source-audit command in `inherited/audit11.py` also requires `Harmonized CHARLS/H_CHARLS_D_do_file.zip` (SHA-256 `ebda374d0dbf1f192f1617f5c5cb4624126eb2fae41385b29365dc8c93845199`) and `2011 CHARLS Wave1/CHARLS2011________.doc` (SHA-256 `689a7365fa93ae7996f2c352df53d4b4254edad29efe936f0f343d834828bbb1`) from the corresponding official distribution, plus the historical project archives. The main analysis imports reusable audit functions and does not invoke that standalone command. The extra documentation files and the original archives are not required for the main entry point.

The project variable dictionary and input lock accompany the code under `analysis/code/reference`. For the analysis entry point, obtain the manuscript's Additional file 4 (`CHARLS_PainSleepFalls_Reproducibility_Inputs_v1_0.zip`) and install these two frozen aggregate verification records using its README and `verify_install.py`, retaining their relative paths:

- `analysis/code/reference/v121/01_patch/analysis_invariance.json`
- `analysis/code/reference/v21/08_QA_AND_REPRODUCIBILITY/nonfit_summary.json`

These two records are required by the cohort and design checks; do not omit them or disable those checks. Additional file 4 contains nine frozen input files in total: these two records, six figure CSVs and the figure-source map. It does not contain the original participant-level data, which must be acquired separately under the provider's access terms.

Obtain the provider's codebook, questionnaire and construction script directly.
