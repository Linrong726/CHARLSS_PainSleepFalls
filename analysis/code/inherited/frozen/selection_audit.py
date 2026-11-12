"""B/W1 retention and selection diagnostics. Return aggregate records only."""

from math import sqrt

import pandas as pd


GROUPS = ("both_persist", "sleep_only", "pain_only", "both_resolve")
M1_FIELDS = ("age", "sex", "education_R", "rural", "prior_fall")
EXTENDED_FIELDS = ("mood9", "chronic12", "walk100", "vision", "memory")
CONTINUOUS_FIELDS = ("age", "mood9", "chronic12", "vision", "memory")
CATEGORICAL_FIELDS = {
    "sex": (1, 2),
    "education": (0, 1, 2),
    "education_R": (0, 1, 2),
    "bd001": tuple(range(1, 12)),
    "rural": (0, 1),
    "prior_fall": (0, 1),
    "walk100": (0, 1),
    "vision": (1, 2, 3, 4, 5),
    "memory": (1, 2, 3, 4, 5),
}
LABELS = {
    "age": "基线年龄（整岁）", "sex": "基线性别", "education": "旧教育三分类",
    "education_R": "修订教育三分类", "bd001": "基线教育原始回答",
    "rural": "基线城乡", "prior_fall": "2011既往跌倒", "mood9": "去睡眠项的9题情绪得分",
    "chronic12": "12项躯体慢病数", "walk100": "步行100米困难", "vision": "远视力顺序评分",
    "memory": "主观记忆评价",
}


def t1_missing_reasons(q):
    """Explain missing T1 classification without altering the original eligibility mask."""
    demo = q.in_demo1.eq(True).fillna(False)
    health = q.in_health1.eq(True).fillna(False)
    proxy = q.proxy1.eq(True).fillna(False)
    unknown = q.proxy_unknown1.eq(True).fillna(False)
    sleep_module = q.in_sleep_module1.eq(True).fillna(False)
    pain_missing, sleep_missing = q.p1.isna(), q.s1.isna()
    # v1.2.1: tracked status was joined to the complete baseline, not just W3 rows.
    tracking = q.t1_tracking_reason.astype("str")
    missing_interview = ~demo
    assert not (q.B_class & missing_interview).any(), "SELECTION_CLASSIFIED_WITHOUT_T1_INTERVIEW"
    assert not (tracking.loc[missing_interview].isna()
                | tracking.loc[missing_interview].eq("")).any(), "SELECTION_T1_TRACKING_REASON_MISSING"

    combined = pd.Series("CLASSIFIABLE", index=q.index, dtype="str")
    combined.loc[pain_missing] = "PAIN_ITEM_UNAVAILABLE"
    combined.loc[sleep_missing] = "SLEEP_ITEM_UNAVAILABLE"
    combined.loc[pain_missing & sleep_missing] = "BOTH_EXPOSURE_ITEMS_UNAVAILABLE"
    combined.loc[sleep_missing & ~sleep_module] = "SLEEP_MODULE_UNAVAILABLE"
    combined.loc[unknown] = "PROXY_STATUS_UNCERTAIN"
    combined.loc[proxy] = "PROXY_EXCLUDED"
    combined.loc[~health] = "HEALTH_MODULE_UNAVAILABLE"
    combined.loc[missing_interview] = tracking.loc[missing_interview]
    # An attended record invalidated upstream solely by a source conflict must
    # remain auditable; otherwise preserve the existing item/proxy precedence.
    unexplained_conflict = ~q.B_class & combined.eq("CLASSIFIABLE") & q.t1_source_conflict
    combined.loc[unexplained_conflict] = "SOURCE_CONFLICT"
    combined.loc[q.B_class] = "CLASSIFIABLE"

    item_reasons = []
    for kind, value in (("pain", q.p1), ("sleep", q.s1)):
        raw = q[f"{kind}_code1"].astype("str")
        code_status = pd.Series("ITEM_BLANK_OR_UNDIFFERENTIATED", index=q.index, dtype="str")
        code_status.loc[raw.isin([".d", "8", "997"])] = "DONT_KNOW"
        code_status.loc[raw.isin([".r", "9", "999"])] = "REFUSAL"
        allowed = [1, 2] if kind == "pain" else [1, 2, 3, 4]
        numeric = pd.to_numeric(raw, errors="coerce")
        code_status.loc[numeric.notna() & ~numeric.isin(allowed + [8, 9, 997, 999])] = "INVALID_ITEM_CODE"
        if kind == "sleep":
            code_status.loc[~sleep_module] = "SLEEP_MODULE_UNAVAILABLE"
        code_status.loc[unknown] = "PROXY_STATUS_UNCERTAIN"
        code_status.loc[proxy] = "PROXY_EXCLUDED"
        code_status.loc[~health] = "HEALTH_MODULE_UNAVAILABLE"
        code_status.loc[missing_interview] = tracking.loc[missing_interview]
        code_status.loc[value.notna() & demo & health] = "VALID"
        item_reasons.append(code_status)
    return combined, *item_reasons


def continuous_group(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    n = len(values)
    return {
        "n": n, "missing": len(series) - n,
        "mean": float(values.mean()) if n else None,
        "sd": float(values.std()) if n >= 2 else None,
        "median": float(values.median()) if n else None,
        "q1": float(values.quantile(.25)) if n else None,
        "q3": float(values.quantile(.75)) if n else None,
    }


def categorical_group(series, levels, level):
    values = pd.to_numeric(series, errors="coerce")
    known = values.isin(levels)
    n = int(known.sum())
    count = int(values.eq(level).sum())
    return {
        "n": n, "missing": len(series) - n,
        "count": count, "proportion": count / n if n else None,
    }


def standardized_difference(retained, excluded, kind):
    if kind == "continuous":
        if min(retained["n"], excluded["n"]) < 2:
            return None, "INSUFFICIENT_NONMISSING_FOR_SAMPLE_SD"
        first, second = retained["mean"], excluded["mean"]
        variance = (retained["sd"] ** 2 + excluded["sd"] ** 2) / 2
    else:
        if min(retained["n"], excluded["n"]) == 0:
            return None, "NO_NONMISSING_IN_ONE_OR_BOTH_GROUPS"
        first, second = retained["proportion"], excluded["proportion"]
        variance = (first * (1 - first) + second * (1 - second)) / 2
    if variance == 0:
        return None, "ZERO_POOLED_VARIANCE"
    return (first - second) / sqrt(variance), "CALCULATED"


def describe_stage(stage, q, keep):
    kept, lost = q.loc[keep], q.loc[~keep]
    comparisons = []

    def add(field, kind, level, retained, excluded):
        smd, status = standardized_difference(retained, excluded, kind)
        comparisons.append({
            "stage": stage, "field": field, "label": LABELS[field], "statistic_type": kind,
            "level": level, "retained_denominator": len(kept), "excluded_denominator": len(lost),
            **{f"retained_{k}": v for k, v in retained.items()},
            **{f"excluded_{k}": v for k, v in excluded.items()},
            "smd_retained_minus_excluded": smd, "abs_smd": abs(smd) if smd is not None else None,
            "smd_status": status,
        })

    for field in CONTINUOUS_FIELDS:
        add(field, "continuous", None, continuous_group(kept[field]), continuous_group(lost[field]))
    for field, levels in CATEGORICAL_FIELDS.items():
        for level in levels:
            add(field, "category_level", level, categorical_group(kept[field], levels, level),
                categorical_group(lost[field], levels, level))
    return comparisons


def aggregate_reasons(stage, subcohort, reasons, **labels):
    denominator = len(subcohort)
    return [
        {"stage": stage, **labels, "reason": str(reason), "n": int(n),
         "denominator": denominator, "fraction": int(n) / denominator}
        for reason, n in reasons.loc[subcohort.index].value_counts(dropna=False).items()
    ]


def audit_selection(d):
    """Compute the three distinct W1 denominators and retained/excluded diagnostics."""
    assert d.index.is_unique, "SELECTION_NONUNIQUE_PERSON_KEY"
    baseline = d.loc[d.B_start].copy()
    classified = baseline.loc[baseline.B_class]
    valid_y = classified.loc[classified.Y.notna()]
    assert baseline.Y.dropna().isin([0, 1]).all(), "SELECTION_NONBINARY_Y"
    assert (baseline.Y.notna() == baseline.outcome_status.eq("VALID_Y")).all(), "SELECTION_Y_STATUS_MISMATCH"
    assert d.B_class.sum() == len(classified), "SELECTION_CLASS_OUTSIDE_B_START"
    assert classified.Bgroup.isin(GROUPS).all(), "SELECTION_UNKNOWN_B_GROUP"

    stage1 = "B_START_TO_T1_CLASSIFIABLE"
    stage2 = "T1_CLASSIFIABLE_TO_VALID_Y"
    stage3 = "VALID_Y_TO_EXTENDED_FIELDS_COMPLETE"
    stages = (
        (stage1, baseline, baseline.B_class),
        (stage2, classified, classified.Y.notna()),
        (stage3, valid_y, valid_y.M2_complete),
    )
    flow, descriptions = [], []
    for stage, q, keep in stages:
        n = int(keep.sum())
        flow.append({"stage": stage, "denominator": len(q), "retained": n,
                     "excluded": len(q) - n, "retention": n / len(q) if len(q) else None})
        descriptions.extend(describe_stage(stage, q, keep))

    cells = []
    for t1_classifiable in (True, False):
        for y_available in (True, False):
            q = baseline.loc[baseline.B_class.eq(t1_classifiable) & baseline.Y.notna().eq(y_available)]
            cells.append({"t1_classifiable": t1_classifiable, "y_available": y_available, "n": len(q),
                          "events": int(q.Y.eq(1).sum()) if y_available else None,
                          "non_events": int(q.Y.eq(0).sum()) if y_available else None})

    t1_status, pain_status, sleep_status = t1_missing_reasons(baseline)
    t1_excluded = baseline.loc[~baseline.B_class]
    assert not t1_status.loc[t1_excluded.index].eq("CLASSIFIABLE").any(), "UNEXPLAINED_T1_CLASSIFICATION_EXCLUSION"
    t1_with_y = t1_excluded.loc[t1_excluded.Y.notna()]
    t1_without_y = t1_excluded.loc[t1_excluded.Y.isna()]
    reasons = []
    for name, excluded in (("ALL_T1_EXCLUDED", t1_excluded), ("VALID_Y", t1_with_y),
                           ("Y_UNAVAILABLE", t1_without_y)):
        reason_sets = {"joint": t1_status, "pain": pain_status, "sleep": sleep_status}
        for definition, status in reason_sets.items():
            rows = aggregate_reasons(stage1, excluded, status, subset=name, reason_type=definition)
            assert sum(x["n"] for x in rows) == len(excluded), "SELECTION_T1_REASON_SUM"
            reasons.extend(rows)
    no_t1_interview = t1_excluded.loc[~t1_excluded.in_demo1.eq(True).fillna(False)]
    t1_tracking_rows = aggregate_reasons(stage1, no_t1_interview,
                                         baseline.t1_tracking_status,
                                         subset="ALL_NO_T1_DEMOGRAPHICS_INTERVIEW",
                                         reason_type="relinked_tracking_status")
    assert sum(r["n"] for r in t1_tracking_rows) == len(no_t1_interview), "SELECTION_T1_TRACKING_SUM"
    reasons.extend(t1_tracking_rows)

    t2_excluded = classified.loc[classified.Y.isna()]
    stage2_reasons = aggregate_reasons(stage2, t2_excluded, classified.outcome_status,
                                       subset="Y_UNAVAILABLE", reason_type="outcome_status")
    assert sum(x["n"] for x in stage2_reasons) == len(t2_excluded), "SELECTION_T2_REASON_SUM"
    reasons.extend(stage2_reasons)
    if "iwstat2" in t2_excluded:
        nonresponse = t2_excluded.loc[t2_excluded.outcome_status.eq("NO_INTERVIEW_OR_UNKNOWN_DISPOSITION")]
        subtype = nonresponse.iwstat2.map({0: "NOT_IN_WAVE", 4: "NONRESPONSE_KNOWN_ALIVE",
                                          9: "NONRESPONSE_VITAL_UNKNOWN"}).fillna("NO_RELEASED_SUBTYPE")
        reasons.extend(aggregate_reasons(stage2, nonresponse, subtype,
                                         subset="NO_INTERVIEW_OR_UNKNOWN_DISPOSITION",
                                         reason_type="release_tracking_status"))

    # Explain missing expanded inputs as mutually exclusive patterns and field marginals.
    fields = list(M1_FIELDS + EXTENDED_FIELDS)
    missing = valid_y[fields].isna()
    assert (valid_y.M2_complete.eq(~missing.any(axis=1))).all(), "SELECTION_M2_COMPLETE_MASK_MISMATCH"
    m2_excluded = valid_y.loc[~valid_y.M2_complete]
    patterns = missing.loc[m2_excluded.index].apply(lambda row: "+".join(row.index[row]), axis=1)
    m2_patterns = aggregate_reasons(stage3, m2_excluded, patterns,
                                   subset="M2_INCOMPLETE", reason_type="missing_pattern")
    assert sum(x["n"] for x in m2_patterns) == len(m2_excluded), "SELECTION_M2_REASON_SUM"
    reasons.extend(m2_patterns)
    for field, n in missing.sum().items():
        reasons.append({"stage": stage3, "subset": "VALID_Y", "reason_type": "missing_field_marginal",
                        "reason": field, "n": int(n), "denominator": len(valid_y),
                        "fraction": int(n) / len(valid_y) if len(valid_y) else None})

    outcome_by_group, outcome_reasons = [], []
    for group in GROUPS:
        q = classified.loc[classified.Bgroup.eq(group)]
        available = int(q.Y.notna().sum())
        outcome_by_group.append({"Bgroup": group, "t1_classifiable": len(q), "y_observed": available,
                                 "events": int(q.Y.eq(1).sum()), "non_events": int(q.Y.eq(0).sum()),
                                 "y_missing": len(q) - available,
                                 "observed_fraction": available / len(q) if len(q) else None})
        for reason, n in q.loc[q.Y.isna(), "outcome_status"].value_counts().items():
            outcome_reasons.append({"Bgroup": group, "reason": str(reason), "n": int(n),
                                    "group_denominator": len(q), "unobserved_denominator": len(q) - available,
                                    "fraction_of_group": int(n) / len(q),
                                    "fraction_of_unobserved": int(n) / (len(q) - available)})

    observed_from_b_start = int(baseline.Y.notna().sum())
    assert sum(x["n"] for x in cells) == len(baseline), "SELECTION_CROSS_TABLE_SUM"
    assert sum(x["n"] for x in cells if x["y_available"]) == observed_from_b_start, "SELECTION_Y_AVAILABLE_SUM"
    assert len(valid_y) + len(t1_with_y) == observed_from_b_start, "SELECTION_Y_OUTSIDE_CLASSIFIED_SUM"
    assert sum(x["y_observed"] for x in outcome_by_group) == len(valid_y), "SELECTION_B_GROUP_OBSERVED_SUM"
    assert sum(x["y_missing"] for x in outcome_by_group) == len(t2_excluded), "SELECTION_B_GROUP_UNOBSERVED_SUM"
    assert sum(x["n"] for x in outcome_reasons) == len(t2_excluded), "SELECTION_B_GROUP_REASON_SUM"

    tables = {
        "selection_flow": flow,
        "selection_t1_by_y": cells,
        "selection_exclusion_reasons": reasons,
        "selection_baseline_comparisons": descriptions,
        "selection_outcome_by_group": outcome_by_group,
        "selection_outcome_reasons_by_group": outcome_reasons,
    }
    summary = {
        "status": "COMPLETE_DESCRIPTIVE",
        "flow": flow,
        "classifiability_by_y": cells,
        "valid_y_in_all_baseline_dual_positive": observed_from_b_start,
        "valid_y_in_t1_classifiable_b": len(valid_y),
        "valid_y_but_t1_unclassifiable": len(t1_with_y),
        "valid_y_but_t1_unclassifiable_reasons": [x for x in reasons if x["stage"] == stage1
                                                  and x["subset"] == "VALID_Y" and x["reason_type"] == "joint"],
        "t1_primary_exclusion_reasons": [r for r in reasons if r["stage"] == stage1
                                         and r["subset"] == "ALL_T1_EXCLUDED" and r["reason_type"] == "joint"],
        "t1_no_interview_tracking_status": t1_tracking_rows,
        "t1_source_conflicts": {
            "baseline_n": int(baseline.t1_source_conflict.sum()),
            "t1_classified_n": int(classified.t1_source_conflict.sum()),
            "classified_with_valid_y_n": int(valid_y.t1_source_conflict.sum()),
        },
        "missing_y_in_t1_classifiable_reasons": stage2_reasons,
        "outcome_by_group": outcome_by_group,
        "nonresponse_detail": "Release does not reliably distinguish refusal, withdrawal, loss of contact, and not yet interviewed.",
        "smd_definition": "Retained minus excluded, observed baseline values only. Continuous uses sqrt((SD_kept^2+SD_excluded^2)/2); categorical levels use sqrt((p_kept*(1-p_kept)+p_excluded*(1-p_excluded))/2). Undefined values remain null.",
        "reconciled": True,
    }
    return tables, summary


def selection_synthetic_tests():
    """Verify selection denominators and NA-SMD handling on artificial rows."""
    n = 6
    d = pd.DataFrame({
        "B_start": [True] * 5 + [False],
        "B_class": [True, True, True, False, False, False],
        "Bgroup": ["both_persist", "both_resolve", "pain_only", None, None, None],
        "Y": [1., 0., None, 1., None, 0.],
        "outcome_status": ["VALID_Y", "VALID_Y", "DEATH", "VALID_Y",
                           "NO_INTERVIEW_OR_UNKNOWN_DISPOSITION", "VALID_Y"],
        "in_demo1": [True, True, True, True, False, True],
        "in_health1": [True, True, True, True, False, True],
        "in_sleep_module1": [True, True, True, True, False, True],
        "proxy1": [False, False, False, True, False, False],
        "proxy_unknown1": [False] * n,
        "dead1": [0, 0, 0, 0, None, 0],
        "t1_tracking_reason": ["", "", "", "", "DEATH_BY_T1_PREVIOUS_WAVE", ""],
        "t1_tracking_status": ["ATTENDED", "ATTENDED", "ATTENDED", "ATTENDED",
                               "DEATH_BY_T1_PREVIOUS_WAVE", "ATTENDED"],
        "t1_source_conflict": [False] * n,
        "p1": [1., 0., 0., None, None, 1.], "s1": [1., 0., 1., None, None, 0.],
        "pain_code1": ["1", "2", "2", "1", ".", "1"],
        "sleep_code1": ["3", "1", "4", "3", ".", "2"],
        "M2_complete": [True, False, True, True, True, True],
        "age": [65, 66, 70, 74, 80, 64], "sex": [1, 2, 1, 2, 2, 1],
        "bd001": [1, 2, 3, 4, 5, 6], "education": [0, 0, 0, 0, 0, 1],
        "education_R": [0, 1, 1, 1, 2, 2], "rural": [0, 0, 0, 1, 1, 0],
        "prior_fall": [0, 0, 1, 1, 0, 1], "mood9": [2., None, 1., 3., 5., 9.],
        "chronic12": [0, 1, 1, 2, 3, 1], "walk100": [0, 0, 0, 1, 1, 0],
        "vision": [2, 2, 2, 3, 5, 1], "memory": [1, 1, 2, 2, 4, 1],
    })
    tables, audit = audit_selection(d)
    assert [r["n"] for r in tables["selection_t1_by_y"]] == [2, 1, 1, 1]
    assert audit["valid_y_in_all_baseline_dual_positive"] == 3
    assert audit["valid_y_in_t1_classifiable_b"] == 2
    assert audit["valid_y_but_t1_unclassifiable_reasons"][0]["reason"] == "PROXY_EXCLUDED"
    assert [r["retained"] for r in tables["selection_flow"]] == [3, 2, 1]
    assert {r["reason"]:r["n"] for r in audit["t1_primary_exclusion_reasons"]} == {
        "PROXY_EXCLUDED": 1, "DEATH_BY_T1_PREVIOUS_WAVE": 1}
    assert sum(r["n"] for r in audit["t1_no_interview_tracking_status"]) == 1
    conflict_case = d.iloc[[0]].copy()
    conflict_case["B_class"] = False
    conflict_case["t1_source_conflict"] = True
    reason, _, _ = t1_missing_reasons(conflict_case)
    assert reason.iloc[0] == "SOURCE_CONFLICT"
    assert sum(r["n"] for r in tables["selection_outcome_reasons_by_group"]) == 1
    smd, status = standardized_difference({"n": 3, "proportion": 0.}, {"n": 2, "proportion": 0.}, "category_level")
    assert smd is None and status == "ZERO_POOLED_VARIANCE"
    return [
        {"phase": "SYNTHETIC", "test": "SELECTION_DENOMINATORS", "status": "PASS",
         "evidence": "Valid followup Y without classifiable T1 is retained in the broad denominator only."},
        {"phase": "SYNTHETIC", "test": "SELECTION_FLOW_AND_REASONS_RECONCILE", "status": "PASS",
         "evidence": "All three sequential retention flags and the explicit death disposition reconcile."},
        {"phase": "SYNTHETIC", "test": "SELECTION_UNDEFINED_SMD_NULL", "status": "PASS",
         "evidence": "Zero pooled variance yields null standardized difference with explicit reason."},
    ]


