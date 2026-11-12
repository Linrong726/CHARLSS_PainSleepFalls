
"""Diagnose historical A and audit the frozen B education repair without refitting A."""
import numpy as np
import pandas as pd
from patsy import dmatrices

OLD_MAPPING = {k: 0 if k <= 5 else 1 if k <= 7 else 2 for k in range(1, 12)}
APPROVED_MAPPING = {k: 0 if k == 1 else 1 if k <= 4 else 2 for k in range(1, 12)}
RAW_LEVELS = [str(k) for k in range(1, 12)]
MISSING_LEVELS = ["SYSTEM_MISSING", "DONT_KNOW", "REFUSAL", "OTHER_EXTENDED_MISSING", "ILLEGAL"]
A_LEVELS = ["11", "10", "01", "00"]
B_LEVELS = ["both_persist", "sleep_only", "pain_only", "both_resolve"]
ANOMALY_TERM = "C(education)[T.2.0]"


def education_revision(raw_bd001, mapping):
    """Map the source BD001 codes only, leaving missing/invalid answers missing."""
    numeric_mapping = {int(k): v for k, v in mapping.items()}
    if numeric_mapping != APPROVED_MAPPING:
        raise ValueError("Education mapping differs from the fixed approved BD001 recoding")
    return pd.to_numeric(raw_bd001, errors="coerce").map(numeric_mapping).astype(float)


def _raw_levels(raw):
    source = raw.astype("string")
    numeric = pd.to_numeric(source, errors="coerce")
    legal = numeric.isin(APPROVED_MAPPING)
    category = pd.Series("ILLEGAL", index=raw.index, dtype="string")
    category.loc[source.str.fullmatch(r"\.[a-z]", na=False)] = "OTHER_EXTENDED_MISSING"
    category.loc[source.isna() | source.isin([".", ""])] = "SYSTEM_MISSING"
    category.loc[source.eq(".d").fillna(False)] = "DONT_KNOW"
    category.loc[source.eq(".r").fillna(False)] = "REFUSAL"
    category.loc[legal] = numeric.loc[legal].astype(int).astype(str)
    return category


def _class_levels(s):
    return s.map({0: "0", 1: "1", 2: "2"}).astype("string").fillna("MISSING")


def _counts(s):
    return dict(N=len(s), events=int(s.Y.eq(1).sum()), non_events=int(s.Y.eq(0).sum()),
                missing_Y=int(s.Y.isna().sum()), communities=int(s.cluster.nunique()),
                missing_community=int(s.cluster.isna().sum()))


def _same_values(a, b):
    return (a.eq(b) | (a.isna() & b.isna())).fillna(False)


def _historic_boundary(history_a, historical_rows):
    rows = [r for r in historical_rows if r["model"] == "M1" and r["term"]]
    if not rows or any(r["design"] != "A" or r["window"] != "W1" for r in rows):
        raise ValueError("Expected original v1.1 historical A/W1 M1 coefficient rows")
    archived = rows[0]
    coefficients = {r["term"]: float(r["coefficient"]) for r in rows}
    counts = _counts(history_a)
    matches = all(counts[k] == int(float(archived[k]))
                  for k in ["N", "events", "non_events"])
    matches &= counts["communities"] == int(float(archived["clusters"]))
    old2 = history_a.loc[history_a.education.eq(2)]
    summary = dict(
        historical_model="v1.1 A/W1 M1",
        refitted=False,
        historical_counts_reconciled=bool(matches),
        archived_term=ANOMALY_TERM,
        archived_coefficient=coefficients.get(ANOMALY_TERM),
        archived_converged=archived["converged"],
        archived_iterations=int(float(archived["iterations"])),
        legacy_level_2_counts=_counts(old2),
    )
    if not matches or ANOMALY_TERM not in coefficients:
        summary.update(status="UNRESOLVED_ARCHIVE_RECONCILIATION", finite_mle=None)
        return summary, []

    # Rebuild the original formula matrix, then evaluate the archived coefficient
    # vector. This is a likelihood profile, with every other coefficient fixed.
    ready = history_a.assign(Agroup=pd.Categorical(history_a.Agroup, categories=A_LEVELS))
    y, x = dmatrices(archived["formula"], ready, NA_action="raise", return_type="dataframe")
    rank = int(np.linalg.matrix_rank(x.to_numpy()))
    same_columns = (len(rows) == len(x.columns) == len(coefficients) and
                    set(x.columns) == set(coefficients))
    same_shape = (x.shape[1] == int(float(archived["p"])) and
                  rank == int(float(archived["rank"])))
    summary.update(p=x.shape[1], rank=rank, archived_formula_columns_match=same_columns,
                   archived_formula_shape_match=same_shape)
    if not (same_columns and same_shape and len(old2)):
        summary.update(status="UNRESOLVED_FORMULA_RECONCILIATION", finite_mle=None)
        return summary, []

    expected_dummy = ready.loc[x.index, "education"].eq(2).to_numpy()
    j = x.columns.get_loc(ANOMALY_TERM)
    is_dummy = bool(np.array_equal(x.iloc[:, j].to_numpy(), expected_dummy))
    summary["archived_term_equals_level_2_indicator"] = is_dummy
    if not is_dummy or old2.Y.eq(1).any():
        summary.update(status="DUMMY_RECESSION_NOT_ESTABLISHED", finite_mle=None)
        return summary, []

    beta = np.array([coefficients[c] for c in x.columns])
    eta = x.to_numpy() @ beta
    yy = y.to_numpy().ravel()
    mu_level = float(np.exp(eta[expected_dummy]).sum())
    ll_at_archived = float((yy * eta - np.exp(eta)).sum())  # log(0!) = log(1!) = 0
    profile = [
        dict(model="historical A M1", term=ANOMALY_TERM,
             delta_from_archived_coefficient=delta, coefficient=beta[j] + delta,
             sum_fitted_means_level_2=mu_level * np.exp(delta),
             score_level_2=-mu_level * np.exp(delta),
             log_likelihood_gain=-mu_level * np.expm1(delta),
             poisson_log_likelihood=ll_at_archived - mu_level * np.expm1(delta))
        for delta in (0.0, -1.0, -5.0, -10.0)
    ]
    summary.update(
        status="ZERO_EVENT_DUMMY_PROVES_NO_FINITE_POISSON_MLE",
        finite_mle=False,
        score_at_archived_coefficient=-mu_level,
        log_likelihood_gain_minus_10=profile[-1]["log_likelihood_gain"],
        proof=("All records in the archived treatment-coded education=2 column have Y=0. "
               "At every finite coefficient vector its score is the negative sum of positive "
               "fitted means in that level. Decreasing that coefficient alone strictly increases "
               "the Poisson log-likelihood while all other observations remain unchanged."),
        interpretation=("The extreme finite archived coefficient is compatible with IRLS stopping "
                        "near a boundary. Full rank, finite reported confidence limits and the "
                        "convergence flag did not establish a finite solution."),
        established_exclusions=[
            "Archived design-matrix rank deficiency is not the explanation.",
            "A coding mismatch between 2011 BD001 and the historical three-level mapping is not required.",
        ],
        unresolved=["Whether any additional recession direction exists in historical A was not assessed."],
    )
    return summary, profile


def audit_education(d, cfg, historical_rows):
    """Return aggregate CSV row sets and a JSON-safe diagnosis for B/W1 and history."""
    revision = cfg["education_revision"]
    labels = revision["dta_value_labels"]
    if set(labels) != set(RAW_LEVELS):
        raise ValueError("The frozen DTA education value labels must cover BD001 codes 1–11")
    raw = _raw_levels(d.bd001)
    numeric = pd.to_numeric(d.bd001, errors="coerce")
    legacy = numeric.map(OLD_MAPPING)
    revised = education_revision(d.bd001, revision["mapping"])
    legacy_ok = _same_values(legacy, d.education)
    revised_ok = _same_values(revised, d.education_R)
    common_m1 = d[["age", "sex", "rural", "prior_fall"]].notna().all(axis=1)
    common_m2 = common_m1 & d[["mood9", "chronic12", "walk100", "vision", "memory"]].notna().all(axis=1)
    base_b = d.B_class & d.Y.notna() & d.cluster.notna()
    old_b_m1 = base_b & d.M1_complete
    old_b_m2 = base_b & d.M2_complete
    new_b_m1 = base_b & common_m1 & revised.notna()
    new_b_m2 = base_b & common_m2 & revised.notna()
    cohort_masks = [
        ("historical_A_M1", d.A_class & d.Y.notna() & d.M1_complete & d.cluster.notna(), "Agroup", A_LEVELS),
        ("B_M1_R", new_b_m1, "Bgroup", B_LEVELS),
        ("B_M2_R_complete", new_b_m2, "Bgroup", B_LEVELS),
    ]
    level_rows = []
    cross_rows = []
    audit_rows = []
    scopes = [(name, mask) for name, mask, _, _ in cohort_masks] + [
        ("historical_A_validY_before_complete_filter", d.A_class & d.Y.notna()),
        ("B_validY_before_complete_filter", d.B_class & d.Y.notna()),
        ("B_t0_double_positive", d.B_start),
        ("W1_baseline_age60", d.age.ge(60)),
        ("W1_all_baseline_demographic_interviews", pd.Series(True, index=d.index)),
    ]
    for name, mask in scopes:
        audit_rows.append(dict(sample=name, N=int(mask.sum()),
            raw_BD001_missing=int((mask & raw.isin(MISSING_LEVELS[:-1])).sum()),
            raw_BD001_illegal=int((mask & raw.eq("ILLEGAL")).sum()),
            legacy_missing=int((mask & d.education.isna()).sum()),
            revised_missing=int((mask & d.education_R.isna()).sum()),
            legacy_mapping_mismatches=int((mask & ~legacy_ok).sum()),
            revised_mapping_mismatches=int((mask & ~revised_ok).sum())))

    scale_specs = [
        ("BD001", raw, RAW_LEVELS + MISSING_LEVELS, labels),
        ("legacy_education", _class_levels(d.education), ["0", "1", "2", "MISSING"],
         {"0": "BD001 1–5", "1": "BD001 6–7", "2": "BD001 8–11"}),
        ("education_R", _class_levels(d.education_R), ["0", "1", "2", "MISSING"],
         {"0": "BD001 1", "1": "BD001 2–4", "2": "BD001 5–11"}),
    ]
    for name, mask, exposure, exposure_levels in cohort_masks:
        for scale, category, levels, scale_labels in scale_specs:
            for level in levels:
                g = d.loc[mask & category.eq(level)]
                row = dict(sample=name, scale=scale, level=level,
                           source_label=scale_labels.get(level, level))
                level_rows.append(row | _counts(g))
                for group in exposure_levels:
                    cross_rows.append(row | dict(exposure=exposure, exposure_level=group)
                                      | _counts(g.loc[g[exposure].eq(group)]))
    historical, profile = _historic_boundary(d.loc[cohort_masks[0][1]], historical_rows)
    historical["legacy_mapping_mismatches_in_historical_A"] = audit_rows[0]["legacy_mapping_mismatches"]
    summary = dict(
        historical_anomaly=historical,
        legacy_mapping_mismatches=int((~legacy_ok).sum()),
        revised_mapping_mismatches=int((~revised_ok).sum()),
        revised_vs_legacy_m1_membership_differences=int(old_b_m1.ne(new_b_m1).sum()),
        revised_vs_legacy_m2_membership_differences=int(old_b_m2.ne(new_b_m2).sum()),
        revised_m1_membership_unchanged=old_b_m1.equals(new_b_m1),
        revised_m2_membership_unchanged=old_b_m2.equals(new_b_m2),
        B_M1_R_revised_levels=[r for r in level_rows if r["sample"] == "B_M1_R" and r["scale"] == "education_R"],
        B_M2_R_revised_levels=[r for r in level_rows if r["sample"] == "B_M2_R_complete" and r["scale"] == "education_R"],
    )
    tables = dict(education_level_counts=level_rows, education_exposure_cross=cross_rows,
                  education_mapping_qc=audit_rows, historical_A_education_boundary=profile)
    return tables, summary

