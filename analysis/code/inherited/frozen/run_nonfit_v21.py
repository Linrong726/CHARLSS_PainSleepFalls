"""Read-only B/W1 v2.1 audit. Builds design matrices only and emits aggregates."""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
import json
import os
from pathlib import Path
import sys
from zipfile import ZipFile

sys.dont_write_bytecode = True
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

GROUPS = ("both_persist", "sleep_only", "pain_only", "both_resolve")
LABELS = {"3": "3–4 days", "4": "5–7 days", "other": "other or missing"}
SEVERITY = {"3": 0, "4": 1}
COVARIATE = "sleep_severity_2011"
GAP = "calendar_month_gap"
CENTER = 36
BASE = "Y ~ C(Bgroup, Treatment(reference=\"both_persist\")) + age_c + C(sex) + C(education_R) + C(rural) + C(prior_fall)"
PRIMARY = BASE + " + C(sleep_severity_2011)"
EXTRAS = "mood9 + chronic12 + walk100 + vision + memory"
MODELS = {
    "B_M1_R_v21": (PRIMARY, "B_M1_R", (COVARIATE,), 11),
    "B_M2_R_v21": (PRIMARY + " + " + EXTRAS, "B_M2_R", (COVARIATE,), 16),
    "B_TIME_R_v21": (PRIMARY + " + calendar_month_gap_c", "B_M1_R", (COVARIATE, GAP + "_c"), 12),
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path.name}")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        out = csv.DictWriter(handle, fieldnames=keys)
        out.writeheader()
        out.writerows(rows)


def verified_stage20(baseline):
    manifest = baseline / "SHA256SUMS.txt"
    files = {}
    for line in manifest.read_text(encoding="utf-8-sig").splitlines():
        expected, relative = line.split(maxsplit=1)
        rel = relative.strip()
        item = baseline / rel
        require(item.resolve().is_relative_to(baseline.resolve()) and digest(item) == expected,
                f"HOLD_ACCESS: v2.0 archive member differs: {rel}")
        files[rel] = expected
    archive = baseline / "CHARLS_PRE_FEASIBILITY_2_0_HANDOFF.zip"
    with ZipFile(archive) as z:
        expected = files | {manifest.name: digest(manifest)}
        require(z.testzip() is None and len(z.namelist()) == len(expected)
                and set(z.namelist()) == set(expected), "HOLD_ACCESS: v2.0 ZIP members or CRC differ")
        for name, sha in expected.items():
            with z.open(name) as f:
                require(hashlib.file_digest(f, "sha256").hexdigest() == sha,
                        f"HOLD_ACCESS: v2.0 ZIP member hash differs: {name}")
    return {"entries": len(files), "manifest_sha256": digest(manifest), "zip_sha256": digest(archive)}


def verify_config(config):
    require(config["window"] == [2011, 2015, 2018] and config["design"] == "B/W1",
            "Frozen design or window changed")
    authorization_keys = ["formal_fit_authorized", "formal_analysis_authorized"]
    found = [config[key] for key in authorization_keys if key in config]
    require(found and not any(found), "Formal fitting must remain disabled")
    require(config["model"]["name"] == "B_M1_R_v21", "Primary model differs from specification")
    normalized = lambda formula: " ".join(formula.replace("'", '"').split())
    for name, (formula, _, _, p) in MODELS.items():
        model = config["models"][name]
        require(normalized(model["formula"]) == normalized(formula) and model["expected_parameters"] == p
                and model["offset"] == "none", f"Configuration differs from no-fit audit formula: {name}")
    require(normalized(config["model"]["formula"]) == normalized(PRIMARY),
            "Primary model declarations disagree")


def severity_from_2011(raw_code, eligible, full_proxy, unknown_proxy):
    allowed = eligible & ~full_proxy.eq(True) & ~unknown_proxy.eq(True)
    return raw_code.map(SEVERITY).where(allowed).astype("Int64")


def month_difference(start, finish):
    return (12 * (finish.dt.year - start.dt.year) + finish.dt.month - start.dt.month).astype("Int64")


def end_counts(q):
    return dict(N=len(q), valid_Y=int(q.Y.notna().sum()), events=int(q.Y.eq(1).sum()),
                non_events=int(q.Y.eq(0).sum()), Y_missing=int(q.Y.isna().sum()))


def base_counts(frame, masks, obs):
    flow = []
    cross = []
    stages = {
        "baseline_dual_positive": masks.B_start,
        "t1_classifiable": masks.B_class,
        "t1_unclassifiable": masks.B_start & ~masks.B_class,
        "t2_valid_Y_among_classified": obs,
        "t2_missing_Y_among_classified": masks.B_class & ~obs,
        "primary_complete_case": masks.B_M1_R,
        "extended_complete_case": masks.B_M2_R,
    }
    grades = frame.sleep_code0.where(frame.sleep_code0.isin(SEVERITY), "other")
    for code in LABELS:
        grade = grades.eq(code)
        for stage, mask in stages.items():
            flow.append(dict(DC015_2011=code, grade=LABELS[code], stage=stage, **end_counts(frame.loc[grade & mask])))
        for group in (*GROUPS, "t1_unclassifiable"):
            mask = (masks.B_class & frame.Bgroup.eq(group) if group != "t1_unclassifiable"
                    else masks.B_start & ~masks.B_class)
            cross.append(dict(DC015_2011=code, grade=LABELS[code], Bgroup=group,
                              **end_counts(frame.loc[grade & mask]),
                              N_M1=int((grade & mask & masks.B_M1_R).sum()),
                              N_M2=int((grade & mask & masks.B_M2_R).sum())))
    return flow, cross


def month_summary(q):
    values = q[GAP].dropna().astype(float)
    quartile = values.quantile([.25, .50, .75])
    return dict(valid_Y=len(q), observed_month_gap=len(values), missing_month_gap=len(q)-len(values),
                mean_months=float(values.mean()), sd_months_sample=float(values.std(ddof=1)),
                median_months=float(quartile.loc[.50]), q1_months=float(quartile.loc[.25]),
                q3_months=float(quartile.loc[.75]), IQR_months=float(quartile.loc[.75]-quartile.loc[.25]),
                min_months=int(values.min()), max_months=int(values.max()),
                minimum_possible_days=int((q.date_lo2-q.date_hi1).dt.days.min()),
                maximum_possible_days=int((q.date_hi2-q.date_lo1).dt.days.max()))


def month_outputs(q):
    summaries = [dict(Bgroup=g, **month_summary(q if g == "ALL" else q.loc[q.Bgroup.eq(g)]))
                 for g in ("ALL", *GROUPS)]
    freq = []
    dates = q.assign(t1_month=q.date_lo1.dt.strftime("%Y-%m"), t2_month=q.date_lo2.dt.strftime("%Y-%m"))
    for group in ("ALL", *GROUPS):
        subset = dates if group == "ALL" else dates.loc[dates.Bgroup.eq(group)]
        for field in ("t1_month", "t2_month", GAP):
            freq.extend(dict(Bgroup=group, field=field, level=str(level), n=int(n))
                        for level, n in subset[field].value_counts(sort=False).sort_index().items())
    pair = dates.groupby(["Bgroup", "t1_month", "t2_month", GAP], observed=True, sort=True).size()
    joint = [dict(Bgroup=g, t1_month=t1, t2_month=t2, calendar_month_gap=int(gap), n=int(n))
             for (g, t1, t2, gap), n in pair.items()]
    persistent = q.loc[q.Bgroup.eq("both_persist"), GAP].astype(float)
    resolved = q.loc[q.Bgroup.eq("both_resolve"), GAP].astype(float)
    a = persistent.value_counts(normalize=True)
    b = resolved.value_counts(normalize=True)
    index = a.index.union(b.index)
    overlap = sum(np.minimum(a.reindex(index, fill_value=0), b.reindex(index, fill_value=0)))
    pooled_sd = float(np.sqrt((persistent.var(ddof=1) + resolved.var(ddof=1))/2))
    comparison = {
        "comparison": "both_resolve minus both_persist", "n_resolve": len(resolved), "n_persist": len(persistent),
        "median_difference_months": float(resolved.median()-persistent.median()),
        "mean_difference_months": float(resolved.mean()-persistent.mean()),
        "smd_pooled_sample_variances": float((resolved.mean()-persistent.mean()) / pooled_sd),
        "discrete_overlap_coefficient": float(overlap),
        "common_month_values": "|".join(str(int(x)) for x in sorted(a.index.intersection(b.index))),
        "overlap_method": "Sum over integer months of the minimum of the two empirical relative frequencies.",
        "smd_method": "Difference of means divided by sqrt((SD_resolve^2 + SD_persist^2)/2); SD uses n-1.",
    }
    return summaries, freq, joint, comparison


def rank(x):
    return int(np.linalg.matrix_rank(x))


def diagnostic(x, q, label):
    values = x.to_numpy(dtype=float)
    nonconstant = x.loc[:, x.columns != "Intercept"].to_numpy(dtype=float)
    sd = nonconstant.std(axis=0, ddof=0)
    standardized = ((nonconstant - nonconstant.mean(axis=0)) /
                    np.where(sd == 0, 1, sd))
    normalized = np.column_stack((np.ones(len(values)), standardized))
    pos = q.Y.eq(1).to_numpy()
    neg = q.Y.eq(0).to_numpy()
    p = x.shape[1]
    matrix_rank, positive_rank, negative_rank = rank(values), rank(values[pos]), rank(values[neg])
    cond = float(np.linalg.cond(normalized))
    counts = end_counts(q)
    result = dict(model=label, **counts, parameters=p, rank=matrix_rank,
                  event_row_rank=positive_rank, nonevent_row_rank=negative_rank,
                  communities=int(q.cluster.nunique()), event_communities=int(q.loc[q.Y.eq(1),"cluster"].nunique()),
                  nonevent_communities=int(q.loc[q.Y.eq(0),"cluster"].nunique()),
                  standardized_condition_number=cond if np.isfinite(cond) else None,
                  events_per_parameter=counts["events"]/p,
                  status=("PASS_FULL_RANK_EVENT_ROWS" if matrix_rank == positive_rank == p else
                          "FAIL_RANK" if matrix_rank < p else "REVIEW_EVENT_ROW_RANK"))
    if GAP + "_c" in x:
        col = x.columns.get_loc(GAP + "_c")
        target = values[:, col]
        other = np.delete(values, col, axis=1)
        # VIF as a projection via SVD only. No auxiliary regression estimator is invoked.
        u, s, _ = np.linalg.svd(other, full_matrices=False)
        basis = u[:, s > s[0]*max(other.shape)*np.finfo(float).eps]
        residual = target - basis @ (basis.T @ target)
        total = np.square(target-target.mean()).sum()
        rest = np.square(residual).sum()
        if total > 0 and rest > 0:
            result["calendar_month_gap_vif"] = float(total/rest)
            result["calendar_month_gap_projection_r2"] = float(1-rest/total)
        else:
            result["calendar_month_gap_vif"] = None
            result["calendar_month_gap_projection_r2"] = None
    coverage = []
    for term in x:
        covered = x[term].ne(0)
        coverage.append(dict(model=label, term=term, **end_counts(q.loc[covered]),
                             communities=int(q.loc[covered,"cluster"].nunique())))
    levels = []
    fields = ["Bgroup", "sex", "education_R", "rural", "prior_fall", COVARIATE]
    if label == "B_M2_R_v21":
        fields += ["walk100", "vision", "memory"]
    if label == "B_TIME_R_v21":
        fields += [GAP]
    for field in fields:
        for key in sorted(q[field].unique()):
            counts_level = end_counts(q.loc[q[field].eq(key)])
            levels.append(dict(model=label, field=field, level=str(key), **counts_level,
                               sparse=counts_level["events"] < 5 or counts_level["non_events"] < 5))
    for group in GROUPS:
        for key in (0, 1):
            counts_level = end_counts(q.loc[q.Bgroup.eq(group) & q[COVARIATE].eq(key)])
            levels.append(dict(model=label, field="Bgroup × sleep_severity_2011",
                               level=f"{group} × {key}", **counts_level,
                               sparse=counts_level["events"] < 5 or counts_level["non_events"] < 5))
    return result, coverage, levels


def synthetic_checks(NoFit, FitBlocked):
    from statsmodels.genmod.generalized_linear_model import GLM

    raw = pd.Series(["3", "4", "1", "2", ".", ".r", ".d", "9", "3", "4", "3"])
    eligible = pd.Series([True]*10+[False])
    proxy = pd.Series([False]*8+[True,False,False])
    uncertain = pd.Series([False]*9+[True,False])
    grades = severity_from_2011(raw, eligible, proxy, uncertain)
    require(grades.iloc[:2].tolist() == [0,1] and grades.iloc[2:].isna().all(),
            "Synthetic raw coding or proxy guard failed")
    start = pd.to_datetime(pd.Series(["2015-12-01", "2016-02-01", "2017-01-01", "2017-02-01", None]))
    finish = pd.to_datetime(pd.Series(["2018-01-01", "2018-02-01", "2017-01-01", "2017-01-01", "2018-01-01"]))
    gaps = month_difference(start, finish)
    require(gaps.iloc[:4].tolist() == [25, 24, 0, -1] and pd.isna(gaps.iloc[4]),
            "Synthetic month arithmetic failed")
    with NoFit() as guard:
        try:
            GLM.fit(None)
        except FitBlocked:
            pass
        require(guard.calls == 1, "Synthetic fit API interception failed")
    return [
        {"check": "SYNTHETIC sleep codes/proxy", "status": "PASS", "detail": "3→0, 4→1; missing/other/proxy/ineligible remain missing"},
        {"check": "SYNTHETIC month arithmetic", "status": "PASS", "detail": "Cross-year 25, same-month 0, reversed -1, missing unassigned"},
        {"check": "SYNTHETIC fit guard", "status": "PASS", "detail": "One deliberate API probe blocked before estimator execution"},
    ]



def verify_source_values(inp, frame, masks, obs):
    from audit_core import canonical, exact_index, string_codes, code_number

    harmonized = inp.harmonized()
    bridge_rows = harmonized.loc[harmonized.ID_w1.ne(""), ["ID_w1"]].reset_index()
    bridge = exact_index(bridge_rows,"ID_w1","v2.1 independent bridge",inp.qc)["ID"]
    health = canonical(inp.read(2011, "health_status_and_functioning", ["ID", "dc015"]),
                       2011, bridge, "v2.1 independent 2011 DC015 read", inp.qc)
    start = frame.loc[masks.B_start]
    require(start.index.isin(health.index).all() and
            start.sleep_code0.equals(string_codes(health.dc015.reindex(start.index)).rename("sleep_code0")),
            "Independent 2011 raw DC015 read differs from retained sleep_code0")
    require(not start.proxy0.eq(True).any() and not start.proxy_unknown0.eq(True).any()
            and start.in_demo0.eq(True).all() and start.s0.eq(1).all(),
            "Original baseline self-report/proxy rules do not hold for entry")
    months = {}
    for wave, suffix in ((2015, 1), (2018, 2)):
        raw = inp.read(wave, "sample_infor", ["ID", "iyear", "imonth"])
        id_shape = raw.ID.str.fullmatch(r"[0-9]{12}")
        keyed = canonical(raw.loc[id_shape], wave, bridge, f"v2.1 independent {wave} date read", inp.qc)
        eligible = keyed.reindex(obs.index)
        yy, mm = code_number(eligible.iyear), code_number(eligible.imonth)
        saved = obs[f"date_lo{suffix}"]
        require(yy.notna().all() and mm.isin(range(1, 13)).all() and
                yy.eq(saved.dt.year).all() and mm.eq(saved.dt.month).all(),
                f"Valid Y records differ from actual sample-record date for {wave}")
        months[wave] = (yy, mm)
    yy1, mm1 = months[2015]
    yy2, mm2 = months[2018]
    recomputed = (12*(yy2-yy1) + mm2-mm1).astype("Int64")
    require(recomputed.equals(obs[GAP]), "Calendar gap disagrees with independently read interview year/month")
    return {
        "DC015_source": inp.metadata[(2011,"health_status_and_functioning","dc015")]["path"],
        "DC015_source_label": inp.metadata[(2011,"health_status_and_functioning","dc015")]["label"],
        "independently_matched_baseline_DC015": len(start),
        "independently_matched_t1_and_t2_source_months": len(obs),
        "invalid_raw_months_among_eligible": 0,
    }


def run(parent, raw, baseline, config_path, out):
    import importlib.util

    verified20 = verified_stage20(baseline)
    # The baseline helper only supplies read/verification functions. Its entrypoint is not run.
    helper_path = baseline / "08_ANALYSIS_SPECIFICATION" / "verify_no_fit.py"
    helper_spec = importlib.util.spec_from_file_location("v20_archive_verifier", helper_path)
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    archives = helper.check_archives(parent)
    v121 = Path(archives["1_2_1"]["root"])
    prior = read_json(baseline / "09_DECISION.json")
    past_data = read_json(baseline / "10_QA_AND_REPRODUCIBILITY" / "no_fit_checks.json")
    data_decision = read_json(v121 / "decision.json")
    require(raw.resolve() == (parent/"原始收据"/"CHARLS_原始数据").resolve(),
            "HOLD_ACCESS: --raw must point to the frozen source root")
    require(prior["input_hashes"] == data_decision["input_hashes"], "HOLD_ACCESS: prior frozen input sets disagree")
    before = helper.check_inputs(prior["input_hashes"])
    require(len(before) == 18 and all(row["match"] for row in before), "HOLD_ACCESS: frozen input checksum failed")
    config = read_json(config_path)
    verify_config(config)
    print(f"PASS checksums before execution: {len(before)} frozen inputs, v2.0 {verified20['entries']} manifest members and ZIP, 3 earlier archives", flush=True)

    source = v121 / "source"
    sys.path[:0] = [str(source), str(source / "legacy")]
    from cohort import build_cohort
    from no_fit_validation import (NoFit, FitBlocked, CORE, EXTRA, add_frozen_covariates, derive_masks, make_design, matrix_digest)
    from tracking_patch import link_tracking
    frozen = read_json(source / "legacy" / "config_v1_2.json")
    tests = synthetic_checks(NoFit, FitBlocked)
    with NoFit() as guard:
        data, evidence = build_cohort(raw)
        frame, tracking = link_tracking(add_frozen_covariates(data, frozen), evidence["inputs"])
        masks = derive_masks(frame, frame.t1_confirmed_death | frame.t1_source_conflict)
        require(frame.index.is_unique and masks.index.equals(frame.index),
                "Duplicate or misaligned member IDs after frozen reconstruction")
        require(masks.B_start.equals(frame.B_start) and masks.B_class.equals(frame.B_class),
                "2015 tracking changes established masks")
        stages = dict(age60=int(frame.age.ge(60).sum()), B_start=int(masks.B_start.sum()), B_class=int(masks.B_class.sum()))
        require(stages == past_data["stage_counts"] and tracking["B_start"]["mortality_source_conflicts"] == 0,
                "v2.0 cohort flow or tracking conflict differs")
        obs_mask = masks.B_class & frame.Y.notna()
        require(obs_mask.equals(masks.B_M1_R), "Eligible Y and archived M1 differ")
        old_inputs = {}
        old_members = {}
        for old, fields in (("B_M1_R", CORE), ("B_M2_R", CORE+EXTRA)):
            q, y, x = make_design(frame, masks[old], fields, frozen["model"]["formulas"][old])
            proof = past_data["model_input"][old]
            require(len(q) == proof["N"] and int(q.Y.eq(1).sum()) == proof["events"]
                    and int(q.Y.eq(0).sum()) == proof["non_events"]
                    and int(q.cluster.nunique()) == proof["clusters"] and x.columns.tolist() == proof["columns"]
                    and matrix_digest(x) == proof["matrix_sha256"], f"Historical matrix digest or outcome count differs: {old}")
            old_inputs[old] = {"historical_matrix_sha256_matched": True, "p": x.shape[1], **end_counts(q)}
            old_members[old] = (q.index.copy(), y.copy(), q.Bgroup.copy())

        frame[COVARIATE] = severity_from_2011(frame.sleep_code0, masks.B_start, frame.proxy0, frame.proxy_unknown0)
        require(frame[COVARIATE].notna().equals(masks.B_start), "Illegal or missing baseline grade among eligible participants")
        frame[GAP] = month_difference(frame.date_lo1.where(obs_mask), frame.date_lo2.where(obs_mask))
        frame[GAP+"_c"] = frame[GAP] - CENTER
        obs = frame.loc[obs_mask]
        require(frame[GAP].notna().equals(obs_mask) and obs[GAP].gt(0).all(), "Missing/nonpositive gap or assignment outside B observed Y")
        require(obs[["in_demo1", "in_demo2", "in_health2", "t0_before_t1", "t1_before_t2", "exact_t1_anchor"]].eq(True).all().all()
                and obs.route2.eq(1).all() and obs.new_fall2.isna().all() and obs.last_lo2.eq(obs.date_lo1).all()
                and obs.date_hi0.lt(obs.date_lo1).all() and obs.date_hi1.lt(obs.date_lo2).all()
                and (obs.date_hi2 - obs.date_lo2).dt.days.between(27,30).all(),
                "Eligible calendar dates, valid route or frozen anchor failed")
        scope_check = {
            "valid_Y": len(obs),
            "valid_Y_exact_2015_anchor": int(obs.last_lo2.eq(obs.date_lo1).sum()),
            "valid_Y_strict_t0_t1_t2_order": int((obs.date_hi0.lt(obs.date_lo1) & obs.date_hi1.lt(obs.date_lo2)).sum()),
            "valid_Y_nonpositive_gap": int(obs[GAP].le(0).sum()),
            "same_month": int(obs[GAP].eq(0).sum()),
            "reversed_month": int(obs[GAP].lt(0).sum()),
            "gap_assigned_outside_observed_Y": int((~obs_mask & frame[GAP].notna()).sum()),
            "classified_missing_Y_excluded_from_gap": int((masks.B_class & ~obs_mask).sum()),
            "valid_Y_but_t1_unclassifiable_excluded_from_gap": int((masks.B_start & ~masks.B_class & frame.Y.notna()).sum()),
            "t1_missing_or_invalid": int(obs.date_lo1.isna().sum()),
            "t2_missing_or_invalid": int(obs.date_lo2.isna().sum()),
            "t2_minus_t1_calendar_year_count": {str(k):int(v) for k,v in
                (obs.date_lo2.dt.year-obs.date_lo1.dt.year).value_counts().sort_index().items()},
        }
        require(scope_check["valid_Y_but_t1_unclassifiable_excluded_from_gap"] == 42, "42-case frozen exception count differs")
        scope_check.update(verify_source_values(evidence["inputs"], frame, masks, obs))
        flow, cross = base_counts(frame, masks, obs_mask)
        for g, old_counts in past_data["groups"].items():
            classified = masks.B_class & frame.Bgroup.eq(g)
            c = end_counts(frame.loc[classified])
            require(c["N"] == old_counts["classified"] and c["valid_Y"] == old_counts["N"]
                    and c["events"] == old_counts["events"] and c["non_events"] == old_counts["non_events"],
                    f"Frozen group distribution differs: {g}")
        months, freq, joint, comparison = month_outputs(obs)
        require([months[0]["minimum_possible_days"], months[0]["maximum_possible_days"]] == [913, 1187],
                "Closed-month day bounds differ from historical archive")
        print("PASS original cohort/member/matrix lineage and independent raw DC015/month reads", flush=True)
        print(f"Flow: {stages}; group timing diagnostics: {comparison}", flush=True)
        summaries, coverage, level_rows = {}, [], []
        for name, (formula, old, added, expected_p) in MODELS.items():
            fields = (CORE if old == "B_M1_R" else CORE+EXTRA)+list(added)
            complete = masks[old] & frame[list(added)].notna().all(axis=1)
            require(complete.equals(masks[old]), f"New candidate has a member excluded for missingness: {name}")
            design_source = frame.loc[complete].copy()
            for field in added:
                design_source[field] = design_source[field].astype("int64")
            q, y, x = make_design(design_source, complete.loc[design_source.index], fields, formula)
            old_idx, old_y, old_g = old_members[old]
            require(old_idx.equals(q.index) and y.equals(old_y) and q.Bgroup.equals(old_g),
                    f"Participant order, outcomes or groups changed: {name}")
            if name == "B_TIME_R_v21":
                q[GAP] = frame.loc[q.index, GAP]
            rec, terms, levels = diagnostic(x, q, name)
            require(rec["parameters"] == expected_p, f"Unexpected column count: {name}")
            require(any(term == "C(sleep_severity_2011)[T.1]" for term in x.columns),
                    f"Sleep reference category is not 0: {name}")
            rec.update(formula=formula, columns=list(x.columns), matrix_sha256=matrix_digest(x),
                       ordered_members_outcomes_groups_unchanged=True, new_covariate_missing=0,
                       zero_event_or_nonevent_levels=sum(level["events"] == 0 or level["non_events"] == 0 for level in levels),
                       sparse_levels_lt5_events_or_nonevents=sum(level["sparse"] for level in levels))
            summaries[name] = rec
            coverage.extend(terms)
            level_rows.extend(levels)
            print(f"Design only {name}: N {rec['N']}, p {rec['parameters']}, rank {rec['rank']}, "
                  f"event rank {rec['event_row_rank']}, non-event rank {rec['nonevent_row_rank']}, "
                  f"communities {rec['communities']}, {rec['status']}", flush=True)
        require(summaries["B_M1_R_v21"]["status"] == "PASS_FULL_RANK_EVENT_ROWS",
                "REVISE_REQUIRED: primary design has insufficient rank")
        require(guard.calls == 0, "Model fit API was called")

    after = helper.check_inputs(prior["input_hashes"])
    require(after == before, "Raw sources changed")
    require(archives == helper.check_archives(parent) and verified20 == verified_stage20(baseline),
            "Protected historical archives changed")
    require(verified20["entries"] == 194, "Unexpected v2.0 manifest count")
    checks = [
        ("REAL raw file checksums", "PASS", f"{len(before)} matched v2.0 and v1.2.1 before and after"),
        ("REAL historical archive hashes", "PASS", "v2.0 expanded files, ZIP CRC and member SHA-256 verified; v1.1/v1.2/v1.2.1 protected manifests and ZIP hashes unchanged"),
        ("REAL independently keyed 2011 DC015", "PASS", f"{stages['B_start']} entries match original item; absent/invalid severity 0; no duplicate keys"),
        ("REAL frozen membership and outcome", "PASS", "Old design hashes and group counts match v2.0; candidate members, order, groups, Y checked in memory"),
        ("REAL actual interview months", "PASS", f"{len(obs)} eligible rows: frozen exact anchor, strict closed-month order, actual original year/month, and correct positive gap"),
        ("REAL window definition scope", "PASS", "Only classified valid Y; 42 valid Y outside classified set do not receive gap"),
        ("REAL fit calls", "PASS", "0; no RR/OR/HR/P-value estimated"),
    ]
    tests.extend(dict(check=check, status=status, detail=detail) for check, status, detail in checks)
    result = {
        "stage": "PRE-FEASIBILITY-2.1", "audit_status": "PASS", "fit_calls": guard.calls,
        "synthetic_rejected_fit_calls": 1, "effect_estimates_generated": 0, "individual_rows_exported": 0,
        "primary_analysis_n": int(obs_mask.sum()), "config_sha256": digest(config_path),
        "raw_hashes_checked": len(before), "raw_hashes_unchanged": True, "historical_hashes_unchanged": True,
        "verified_stage20": verified20,
        "prior_archives": {key: {k:v for k,v in value.items() if k not in {"root", "files", "zip"}} for key,value in archives.items()},
        "stage_counts": stages, "old_designs": old_inputs, "models": summaries,
        "sleep_flow": flow, "sleep_by_group": cross, "month_summary": months,
        "time_comparison": comparison, "date_scope_and_lineage": scope_check,
        "retained_frozen_context": {"2015_unknown_survival": 51, "valid_Y_not_classifiable": 42,
                                   "both_resolve_events": 29, "anchor_day_bounds": [913, 1187]},
        "checks": tests,
    }
    out.mkdir(parents=True, exist_ok=True)
    tables = out / "tables"
    qa = out / "08_QA_AND_REPRODUCIBILITY"
    tables.mkdir(exist_ok=True)
    qa.mkdir(exist_ok=True)
    for name, rows in (
        ("sleep_flow",flow), ("sleep_by_group",cross), ("month_summary",months),
        ("month_frequencies",freq), ("month_joint_frequency",joint), ("design_summary",list(summaries.values())),
        ("design_levels",level_rows), ("design_event_coverage",coverage), ("time_comparison",[comparison]),
        ("input_hashes",before), ("checks",tests),
        ("date_scope_and_lineage", [{"check": key, "value": value} for key,value in scope_check.items()]),
    ):
        write_csv(tables/(name+".csv"), rows)
    with (qa / "nonfit_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"PASS source/archives unchanged. Formal fit calls: {guard.calls}. No effect estimates and no participant rows.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("parent", "raw", "baseline", "config", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.is_relative_to(args.baseline.resolve()) and not output.is_relative_to(args.raw.resolve()),
            "Refusing to write under a protected root")
    run(args.parent, args.raw, args.baseline, args.config, output)


if __name__ == "__main__":
    main()




