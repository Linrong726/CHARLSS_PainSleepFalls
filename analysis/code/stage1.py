"""Rebuild the locked CHARLS cohort and run Stage 1 from authorized source files."""

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
from patsy import build_design_matrices
from scipy.special import expit, logit
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "inherited"))
import audit11
import cohort
import formal_models as base
from frozen.legacy import audit_core
from frozen.selection_audit import categorical_group, continuous_group, standardized_difference

GROUP = dict(zip(base.GROUPS, ("G1", "G2", "G3", "G4")))
SOURCE = "CHARLS 2011/2015/2018 and Harmonized D"
REVISION = "STAGE1-v1.0"
POST = "targeted post hoc Stage 1"
FIT_OPTIONS = dict(method="IRLS", scale=1, cov_type="cluster", use_t=False, maxiter=300,
                   tol=1e-12, rtol=0, tol_criterion="deviance")
PROB_TOL = 8 * np.finfo(float).eps
GRAD_TOL = dict(atol=2e-8, rtol=2e-6)
RAW_HEALTH = ["mood9", "chronic12_raw2011", "walk100", "vision", "memory"]


def ensure(test, message):
    if not test:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save(directory, name, rows):
    (rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)).to_csv(
        directory / f"{name}.csv", index=False, encoding="utf-8-sig")


def model_spec():
    old = base.MODELS
    no_walk = [field for field in RAW_HEALTH if field != "walk100"]
    return {
        "MAIN": old["B_M1_R_v21"],
        "S1": old["B_S1_NOPRIOR_v21"],
        "F_BRIDGE": old["B_S2_BRIDGE_v21"],
        "F_HARM": old["B_M2_R_v21"],
        "F_RAW": (base.F + " + " + " + ".join(RAW_HEALTH), "extended", 16, "prior source sensitivity",
                  base.CORE + RAW_HEALTH),
        "S4": old["B_TIME_R_v21"],
        "N1": (base.F + " + " + " + ".join(no_walk), "extended", 15, POST, base.CORE + no_walk),
        "N2": (base.F + " + " + " + ".join(no_walk), "RSTAR", 15, POST, base.CORE + no_walk),
        "N3": (base.F, "RSTAR", 11, POST, base.CORE),
    }


def source_for(fields):
    if "chronic12_raw2011" in fields:
        return "Raw 2011"
    if "chronic12" in fields:
        return "Harmonized D"
    return "None"


def select(frame, main):
    p = frame.B_M1_R
    f = frame.B_M2_R
    extended = [field for field in base.EXTRA if field != "walk100"]
    r0 = p & frame[extended].notna().all(axis=1)
    rstar = r0 & frame.chronic12_raw2011.notna()
    only_walk = r0 & frame.walk100.isna()
    ensure(not (f & ~rstar).any() and not (rstar & ~r0).any() and not (r0 & ~p).any(),
           "Nested membership check failed")
    ensure((r0 & ~f).equals(only_walk), "R0 minus F differs from only-walk-missing")
    ensure((r0 & ~rstar).equals(r0 & frame.chronic12_raw2011.isna()), "R0 minus R* rule")
    ensure(frame.index.is_unique and frame.index.str.fullmatch(r"[0-9]{12}").all(), "Canonical ID error")
    ensure(main.index.equals(frame.index[p]), "Raw audit lost ordered membership")
    return dict(P=p, F=f, R0=r0, RSTAR=rstar, ONLY_WALK=only_walk, RAW_INCOMPLETE=r0 & ~rstar)


def summarize(frame, masks, raw_codes, harm_codes, out, inherited):
    y = frame.Y
    classified = frame.B_class
    starter = frame.B_start
    no_y = classified & y.isna()
    walk = masks["ONLY_WALK"]
    p, f = masks["P"], masks["F"]
    summary = {
        "AGE60": frame.age.ge(60), "B_START": starter, "B_CLASS": classified,
        "UNCLASSIFIED": starter & ~classified, "NO_VALID_Y": no_y, **masks,
        "VALID_Y_UNCLASSIFIED": starter & ~classified & y.notna(), "S3": cohort.s3_eligibility(frame),
        "P_NOT_F": p & ~f, "P_NOT_R0": p & ~masks["R0"],
    }
    save(out, "flow", [dict(stage=name, n=int(mask.sum()),
                            events=int(y.loc[mask].sum()) if name in {*masks, "VALID_Y_UNCLASSIFIED",
                                                                       "P_NOT_F", "P_NOT_R0"} else None,
                            clusters=int(frame.loc[mask, "cluster"].nunique()) if name in masks else None)
                       for name, mask in summary.items()])
    classified_data = frame.loc[classified]
    status = pd.Series("OTHER", index=classified_data.index)
    status.loc[classified_data.outcome_status.eq("DEATH")] = "DEATH"
    ensure(not (status.eq("DEATH") & classified_data.Y.notna()).any(), "Death/outcome conflict")
    status.loc[classified_data.Y.eq(1)] = "FALL"
    status.loc[classified_data.Y.eq(0)] = "NO_FALL"
    followup, counts, details = [], [], []
    for bgroup, group_id in GROUP.items():
        group_status = status.loc[classified_data.Bgroup.eq(bgroup)]
        for label in ("FALL", "NO_FALL", "DEATH", "OTHER"):
            n = int(group_status.eq(label).sum())
            followup.append(dict(group_id=group_id, status=label, n=n,
                                 denominator=len(group_status), percent=100 * n / len(group_status)))
        for name, mask in masks.items():
            q = frame.loc[mask & frame.Bgroup.eq(bgroup)]
            counts.append(dict(sample_id=name, group_id=group_id, n=len(q), events=int(q.Y.sum()),
                               clusters=q.cluster.nunique(), crude_proportion=q.Y.mean() if len(q) else None))
        q = classified_data.loc[classified_data.Bgroup.eq(bgroup)]
        for reason, n in q.loc[q.Y.isna(), "outcome_status"].value_counts().items():
            details.append(dict(group_id=group_id, reason=reason, n=int(n), denominator=len(q)))
    ensure(sum(row["n"] for row in followup) == int(classified.sum()), "Follow-up states do not close")
    save(out, "followup", followup)
    save(out, "followup_detail", details)
    save(out, "sample_groups", counts)
    extra_missing = frame.loc[p, base.EXTRA].isna()
    save(out, "missing", [dict(field=name, n_missing=int(values.sum()), denominator=int(p.sum()))
                          for name, values in extra_missing.items()] +
         [dict(field="only_walk100", n_missing=int(walk.sum()), denominator=int(p.sum()))])
    patterns = extra_missing.apply(lambda row: "+".join(row.index[row]) or "complete", axis=1)
    save(out, "missing_patterns", patterns.value_counts().rename_axis("pattern").reset_index(name="n"))
    missing_walk = raw_codes.loc[extra_missing.index[extra_missing.walk100], list(audit11.WALK_ITEMS)].copy()
    missing_walk["harmonized_walk"] = harm_codes.loc[missing_walk.index, "r1walk100a"]
    save(out, "walk_source", missing_walk.value_counts().rename("n").reset_index())
    diseases = audit11.DISEASES
    fids = frame.index[f]
    h = harm_codes.loc[fids, [row[1] for row in diseases]].to_numpy()
    raw = raw_codes.loc[fids, [row[3] for row in diseases]].replace({"1": "1", "2": "0"}).to_numpy()
    states_differ = np.any(raw != h, axis=1)
    total_differ = frame.loc[f, "chronic12_raw2011"].ne(frame.loc[f, "chronic12"]).to_numpy()
    flagged = harm_codes.loc[fids, [row[2] for row in diseases]].isin(["5", "6"]).any(axis=1).to_numpy()
    ensure(np.all(~total_differ | states_differ), "Different totals require different item states")
    save(out, "chronic", [dict(status=label, n=int(mask.sum()), denominator=len(fids))
                           for label, mask in (("SAME_ALL", ~states_differ),
                                               ("STATUS_DIFF_SAME_TOTAL", states_differ & ~total_differ),
                                               ("TOTAL_DIFF", total_differ), ("DISPUTE_FLAG", flagged))])
    sel = []
    comparisons = {
        "CLASS_vs_UNCLASS": (starter, classified),
        "P_vs_NO_VALID_Y": (classified, p),
        "F_vs_P_NOT_F": (p, f),
        "R0_vs_P_NOT_R0": (p, masks["R0"]),
        "RSTAR_vs_RAW_INCOMPLETE": (masks["R0"], masks["RSTAR"]),
    }
    levels = dict(cohort.BASELINE_LEVELS)
    continuous = [*cohort.BASELINE_CONT, "chronic12_raw2011"]
    for name, (parent, kept) in comparisons.items():
        for field in [*continuous, *levels]:
            # Raw 2011 source counts were linked for P only. Avoid implying source absence outside P.
            if field == "chronic12_raw2011" and (parent & ~p).any():
                continue
            left, right = frame.loc[kept, field], frame.loc[parent & ~kept, field]
            for level in levels.get(field, (None,)):
                kind = "continuous" if level is None else "category_level"
                if level is None:
                    a, b = continuous_group(left), continuous_group(right)
                else:
                    a, b = (categorical_group(values, levels[field], level) for values in (left, right))
                smd, smd_status = standardized_difference(a, b, kind)
                sel.append(dict(comparison=name, field=field, level=level, statistic_type=kind, smd=smd,
                                smd_status=smd_status, retained_total=len(left), excluded_total=len(right),
                                **{f"retained_{k}": v for k, v in a.items()},
                                **{f"excluded_{k}": v for k, v in b.items()}))
    save(out, "selection", sel)
    for name in ("baseline", "s3", "s3_reason_breakdown", "month_summary", "month_frequencies",
                 "selection_exclusion_reasons", "cohort_linkage_qc"):
        data = pd.read_csv(inherited / "cohort_tables" / f"{name}.csv")
        if "Bgroup" in data:
            data["group_id"] = data.Bgroup.map(GROUP).fillna(data.Bgroup)
        save(out, name, data)


def manual_design(frame, model, forced=None):
    """Independent coding path with no Patsy formula evaluation."""
    groups = frame.Bgroup if forced is None else pd.Series(forced, index=frame.index)
    columns = {"Intercept": np.ones(len(frame))}
    for key in tuple(GROUP)[1:]:
        columns[f'C(Bgroup, Treatment(reference="both_persist"))[T.{key}]'] = groups.eq(key)
    for field in ("sex", "education_R", "rural", "prior_fall", "sleep_severity_2011"):
        if model == "S1" and field == "prior_fall":
            continue
        for level in base.LEVELS[field][1:]:
            columns[f"C({field})[T.{level}]"] = frame[field].eq(level)
    columns["age_c"] = frame.age.astype(float) - 65
    if model == "N2":
        for field in ("mood9", "chronic12_raw2011", "vision", "memory"):
            columns[field] = frame[field]
    return pd.DataFrame(columns, index=frame.index).astype(float)


def fit_stage1(frame, masks, models, out, log):
    sample_map = {"main": "P", "extended": "F", "RSTAR": "RSTAR"}
    old, old_primary = base.MODELS, base.PRIMARY
    try:
        base.MODELS = models
        base.PRIMARY = "MAIN"
        designs = {name: base.model_input(frame.loc[masks[sample_map[spec[1]]]], name)
                   for name, spec in models.items()}
        tables = base.run(designs, "STAGE1_REAL", log)
    finally:
        base.MODELS = old
        base.PRIMARY = old_primary
    for name, records in tables.items():
        save(out, f"model_{name}", records)
    ensure(all(row["status"].startswith("PASS") for row in tables["diagnostics"]), "At least one model failed")
    rows = []
    for row in tables["contrasts"]:
        model = row["model"]
        spec = models[model]
        gid = GROUP[row["comparison"]]
        rows.append(dict(sample_id=sample_map[spec[1]], model_id=model, contrast_id=f"{gid}_vs_G1",
                         group_id=gid, n=row["n"], events=row["events"], clusters=row["clusters"],
                         estimate=row["rr"], ci_low=row["ci_low"], ci_high=row["ci_high"], estimand="RR",
                         adjustment=spec[0], chronic_source=source_for(spec[4]), role=row["role"],
                         standardization_reference=None, beta=row["beta"], se=row["se"], p_value=row["p_value"],
                         input_version=SOURCE, execution_version=REVISION))
    spec_rows = [dict(model_id=name, sample_id=sample_map[spec[1]], formula=spec[0], parameters=spec[2],
                      role=spec[3], chronic_source=source_for(spec[4]), family="Poisson", link="log")
                 for name, spec in models.items()]
    crosscheck = {}
    for name in ("MAIN", "N2"):
        q, _, x = designs[name]
        q_original = frame.loc[q.index]
        design = manual_design(q_original, name)
        ensure(design.columns.equals(x.columns) and np.array_equal(design.to_numpy(), x.to_numpy()),
               f"{name} independent matrix mismatch")
        X, y = design.to_numpy(), q.Y.to_numpy()
        beta = np.array([row["beta"] for row in tables["coefficients"] if row["model"] == name])
        cov = np.array([row["covariance"] for row in tables["covariance"]
                        if row["model"] == name]).reshape(len(beta), len(beta))
        mu = np.exp(X @ beta)
        scores = X * (y - mu)[:, None]
        clustered = pd.DataFrame(scores).groupby(q.cluster.to_numpy()).sum().to_numpy()
        n, p = X.shape
        g = len(clustered)
        bread = np.linalg.inv(X.T @ np.diag(mu) @ X)
        correction = g / (g - 1) * (n - 1) / (n - p)
        manual_cov = correction * bread @ (clustered.T @ clustered) @ bread
        tighter = sm.GLM(y, X, family=sm.families.Poisson()).fit(maxiter=300, tol=1e-13, scale=1)
        ensure(tighter.converged and np.allclose(tighter.params, beta, **base.ACCEPTANCE["coefficient"]),
               f"{name} tighter IRLS mismatch")
        ensure(np.allclose(manual_cov, cov, **base.COV_TOL), f"{name} independent covariance mismatch")
        crosscheck[name] = dict(n=n, events=int(y.sum()), clusters=g, parameters=p,
                                design_exact=True, correction=correction,
                                max_cov_abs=float(np.max(np.abs(manual_cov - cov))),
                                max_tight_beta_abs=float(np.max(np.abs(tighter.params - beta))),
                                max_abs_score=float(np.max(np.abs(scores.sum(axis=0)))),
                                fitted_min=float(mu.min()), fitted_max=float(mu.max()))
    return designs, tables, rows, spec_rows, crosscheck


def probabilities(frame, design, tables, out):
    q, _, x = design
    n, p = x.shape
    g = q.cluster.nunique()
    scenarios = {}
    for bgroup, group_id in GROUP.items():
        scenario = q.copy()
        scenario["Bgroup"] = pd.Categorical([bgroup] * n, categories=base.GROUPS)
        xnew = np.asarray(build_design_matrices([x.design_info], scenario, NA_action="raise")[0])
        manual = manual_design(frame.loc[q.index], "MAIN", forced=bgroup).to_numpy()
        ensure(np.array_equal(manual, xnew), f"{group_id} independent scenario differs")
        scenarios[group_id] = xnew
    beta = np.array([r["beta"] for r in tables["coefficients"] if r["model"] == "MAIN"])
    cov = np.array([r["covariance"] for r in tables["covariance"] if r["model"] == "MAIN"]).reshape(p, p)
    diagnostics = []

    def inspect(tag, link, params):
        for gid, matrix in scenarios.items():
            pred = link(matrix @ params)
            finite = np.isfinite(pred)
            diagnostics.append(dict(model_id=tag, group_id=gid, n=n, minimum=float(pred[finite].min()) if finite.any() else None,
                                    maximum=float(pred[finite].max()) if finite.any() else None,
                                    nonfinite=int((~finite).sum()), above_one=int((pred > 1).sum()),
                                    below_zero=int((pred < 0).sum()),
                                    outside_tolerance=int(((pred > 1 + PROB_TOL) | (pred < -PROB_TOL)).sum())))

    model_id, link = "ABS_POISSON", np.exp
    inspect(model_id, link, beta)
    aux = None
    if any(r["nonfinite"] or r["outside_tolerance"] for r in diagnostics):
        model_id, link = "ABS_LOGISTIC", expit
        fit = sm.GLM(q.Y, x, family=sm.families.Binomial()).fit(
            **FIT_OPTIONS, cov_kwds={"groups": q.cluster.to_numpy(), "use_correction": True, "df_correction": True})
        beta, cov = np.asarray(fit.params), np.asarray(fit.cov_params())
        inspect(model_id, link, beta)
        fitted = np.asarray(fit.fittedvalues)
        aux = dict(n=n, events=int(q.Y.sum()), clusters=g, parameters=p, columns=x.columns.tolist(),
                   converged=bool(fit.converged), rank=int(np.linalg.matrix_rank(x)),
                   finite_coefficients=bool(np.isfinite(beta).all()), fitted_min=float(fitted.min()),
                   fitted_max=float(fitted.max()), max_abs_score=float(np.abs(x.to_numpy().T @ (q.Y - fitted)).max()),
                   covariance="2011 community sandwich with CR1 correction")
        ensure(aux["converged"] and aux["rank"] == p and aux["finite_coefficients"] and
               np.isfinite(cov).all() and aux["max_abs_score"] / n < 1e-7, "Auxiliary probability fit invalid")
    save(out, "probability_scenarios", diagnostics)
    ensure(not any(r["nonfinite"] or r["outside_tolerance"] for r in diagnostics if r["model_id"] == model_id),
           "Selected probability predictions outside legitimate range")
    results = []
    meta = dict(sample_id="P", model_id=model_id, n=n, events=int(q.Y.sum()), clusters=g,
                adjustment=base.F, chronic_source="None", role=POST, standardization_reference="P",
                input_version=SOURCE, execution_version=REVISION)
    means, gradients = {}, {}
    for gid, matrix in scenarios.items():
        pred = link(matrix @ beta)
        means[gid] = mean = float(np.mean(pred))
        derivative = pred if model_id == "ABS_POISSON" else pred * (1 - pred)
        gradients[gid] = grad = np.mean(matrix * derivative[:, None], axis=0)
        se = float(np.sqrt(grad @ cov @ grad))
        ci = expit(logit(mean) + np.array([-1, 1]) * base.Z975 * se / (mean * (1 - mean)))
        results.append(dict(**meta, contrast_id=gid, group_id=gid, estimate=mean, ci_low=ci[0], ci_high=ci[1],
                            estimand="PROB", se=se))
    diff = means["G4"] - means["G1"]
    grad = gradients["G4"] - gradients["G1"]
    se = float(np.sqrt(grad @ cov @ grad))
    results.append(dict(**meta, contrast_id="G4_vs_G1", group_id="G4", estimate=diff,
                        ci_low=diff - base.Z975 * se, ci_high=diff + base.Z975 * se, estimand="RD", se=se))

    def direct(gid, params):
        manual = manual_design(frame.loc[q.index], "MAIN", forced=next(k for k, v in GROUP.items() if v == gid))
        return math.fsum(float(link(np.dot(row, params))) for row in manual.to_numpy()) / n

    means_check = {gid: direct(gid, beta) for gid in ("G1", "G4")}
    def rd(params):
        return direct("G4", params) - direct("G1", params)
    step = 1e-6
    finite_grad = np.array([(rd(beta + d) - rd(beta - d)) / (2 * step) for d in np.eye(p) * step])
    finite_var = float(finite_grad @ cov @ finite_grad)
    ensure(all(abs(means[gid] - means_check[gid]) < 1e-13 for gid in means_check) and
           abs(rd(beta) - diff) < 1e-13 and np.allclose(grad, finite_grad, **GRAD_TOL) and
           np.isclose(finite_var, se ** 2, **GRAD_TOL), "Independent probability check failed")
    grad_cov = float(gradients["G1"] @ cov @ gradients["G4"])
    save(out, "probability_covariance", [dict(model_id=model_id, group_1=a, group_2=b,
                                              covariance=float(ga @ cov @ gb))
                                         for a, ga in gradients.items() for b, gb in gradients.items()])
    return results, dict(model_id=model_id, fit=aux, reference="P empirical covariate distribution held fixed",
                         direct=means_check, rd=diff, rd_independent=rd(beta), rd_variance=se ** 2,
                         rd_variance_finite_difference=finite_var, g1_g4_covariance=grad_cov,
                         finite_difference_step=step, max_gradient_abs_difference=float(np.max(abs(grad - finite_grad))),
                         tolerance=GRAD_TOL)


def independent_counts(frame, masks, out, private):
    """Loop over canonical members and rebuild groups, endpoints and nested selections."""
    member = {key: set() for key in masks}
    status = {}
    expected_group = {(1, 1): "G1", (1, 0): "G2", (0, 1): "G3", (0, 0): "G4"}
    events = {key: 0 for key in GROUP.values()}
    p_counts = events.copy()
    f_counts, f_events = events.copy(), events.copy()
    for person, row in frame.loc[frame.B_class].iterrows():
        gid = expected_group[row.p1, row.s1]
        ensure(gid == GROUP[row.Bgroup], "Exposure polarity mismatch")
        valid = pd.notna(row.Y)
        category = ("FALL" if row.Y == 1 else "NO_FALL") if valid else (
            "DEATH" if row.dead2 == 1 else "OTHER")
        status[gid, category] = status.get((gid, category), 0) + 1
        if not valid:
            continue
        ensure(row.in_demo1 and row.in_demo2 and row.in_health2 and row.route2 == 1
               and pd.isna(row.new_fall2) and row.last_lo2 == row.date_lo1
               and row.date_hi0 < row.date_lo1 and row.date_hi1 < row.date_lo2
               and not row.proxy2 and not row.proxy_unknown2 and row.Y == row.fall2,
               "Independent valid outcome chain failed")
        member["P"].add(person)
        p_counts[gid] += 1
        events[gid] += int(row.Y)
        if all(pd.notna(row[key]) for key in ("mood9", "chronic12", "vision", "memory")):
            member["R0"].add(person)
            if pd.notna(row.walk100):
                member["F"].add(person)
                f_counts[gid] += 1
                f_events[gid] += int(row.Y)
            else:
                member["ONLY_WALK"].add(person)
            key = "RSTAR" if pd.notna(row.chronic12_raw2011) else "RAW_INCOMPLETE"
            member[key].add(person)
    ensure(all(values == set(frame.index[masks[key]]) for key, values in member.items()), "Independent ID sets differ")
    official = pd.read_csv(out / "followup.csv")
    ensure(all(r.n == status.get((r.group_id, r.status), 0) for r in official.itertuples()),
           "Independent follow-up counts differ")
    official_groups = pd.read_csv(out / "sample_groups.csv")
    for tag, totals, outcomes in (("P", p_counts, events), ("F", f_counts, f_events)):
        ensure(all(r.n == totals[r.group_id] and r.events == outcomes[r.group_id]
                   for r in official_groups.loc[official_groups.sample_id.eq(tag)].itertuples()),
               f"Independent {tag} group counts differ")
    pd.DataFrame({key: mask.loc[frame.B_start].astype(int) for key, mask in masks.items()}).to_csv(
        private / "LOCAL_MEMBER_AUDIT.csv", index_label="canonical_id", encoding="utf-8-sig")
    return dict(status="PASS", groups={f"{g}:{s}": n for (g, s), n in status.items()},
                membership={name: dict(n=len(ids), sha256=hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest())
                            for name, ids in member.items()},
                P_group_counts=p_counts, P_group_events=events, F_group_counts=f_counts, F_group_events=f_events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New directory for shareable output")
    parser.add_argument("--private", type=Path, required=True, help="New directory for local member audit")
    parser.add_argument("--node", help="Node executable for the workbook exporter")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    root = args.out.resolve()
    private = args.private.resolve()
    ensure(not private.is_relative_to(root) and not root.is_relative_to(private),
           "Private membership directory must be separate from shareable output")
    private.mkdir(parents=True, exist_ok=False)
    out = root / "results"
    log = root / "run"
    out.mkdir(parents=True, exist_ok=False)
    log.mkdir()
    models = model_spec()
    now = datetime.now(timezone(timedelta(hours=8))).isoformat()
    freeze = dict(version=REVISION, timestamp=now, post_hoc=POST, models=models, group_mapping=GROUP,
                  input=SOURCE, key_mapping="Official Harmonized D ID_w1 to 12-character ID, exact one-to-one",
                  r0="P with every original extension field valid except walk100 need not be valid",
                  rstar="R0 with valid raw 2011 count from all 12 doctor-diagnosis items",
                  covariance="2011 community CR1 [G/(G-1)]*[(N-1)/(N-p)]; p includes intercept",
                  weights=None, offset=None, wald="normal; log scale for RR, logit for probability, identity for RD",
                  probability_range_tolerance=PROB_TOL, finite_difference_tolerance=GRAD_TOL,
                  sandwich_tolerance=base.COV_TOL, historical_result_tolerance=base.REPRO_TOL,
                  fit=base.frozen_config()["fit"], logistic_fit=FIT_OPTIONS,
                  code_hashes={p.relative_to(HERE).as_posix(): sha(p) for p in
                               sorted(HERE.rglob("*.py"))})
    write_json(log / "EXECUTION_FREEZE.json", freeze)
    write_json(log / "ENVIRONMENT.json", dict(python=platform.python_version(), os=platform.platform(),
               packages={key: version(key) for key in
                         ("numpy", "pandas", "scipy", "patsy", "statsmodels", "openpyxl", "matplotlib")},
               command=sys.argv, seeds="None. All analyses are deterministic."))
    raw = Path(config["data_root"])
    pins = json.loads((HERE / "reference" / "input_lock.json").read_text(encoding="utf-8"))
    input_evidence = []
    for pin in pins:
        if pin["used_in"] != "analysis":
            continue
        path = raw / pin["file"]
        digest = sha(path)
        ensure(digest == pin["sha256"], f"Raw version mismatch: {pin['file']}")
        input_evidence.append(dict(**pin, checked_sha256=digest))
    write_json(log / "INPUT_HASHES.json", input_evidence)
    inherited = log / "inherited_cohort"
    frame = cohort.build(raw, HERE / "reference", inherited)
    print("Source files rebuilt using the locked cohort", flush=True)
    main_frame, raw_codes, harm_codes, _, linkage = audit11.read_audit_fields(raw, frame, audit_core)
    frame["chronic12_raw2011"] = main_frame.chronic12_raw2011.reindex(frame.index)
    write_json(log / "INDEPENDENT_SOURCE_LINKAGE.json", linkage)
    masks = select(frame, main_frame)
    summarize(frame, masks, raw_codes, harm_codes, out, inherited)
    independent = independent_counts(frame, masks, out, private)
    designs, tables, results, specs, checks = fit_stage1(frame, masks, models, out, log / "FIT_ATTEMPTS.jsonl")
    absolute, probability_checks = probabilities(frame, designs["MAIN"], tables, out)
    results.extend(absolute)
    pm = probability_checks["model_id"]
    specs.append(dict(model_id=pm, sample_id="P", formula=base.F, parameters=11, role=POST, chronic_source="None",
                      family="Poisson" if pm == "ABS_POISSON" else "Binomial",
                      link="log" if pm == "ABS_POISSON" else "logit"))
    save(out, "results_master", results)
    save(out, "model_specifications", specs)
    write_json(log / "INDEPENDENT_NUMERICAL_QA.json", dict(status="PASS", counts=independent, covariance=checks,
                                                         probability=probability_checks))
    print(f"Absolute probabilities completed using {pm}", flush=True)
    if (HERE / "figures_stage1.py").exists():
        subprocess.run([sys.executable, str(HERE / "figures_stage1.py"), str(out), str(root / "figures")], check=True)
    if (HERE / "tables_stage1.py").exists():
        subprocess.run([sys.executable, str(HERE / "tables_stage1.py"), str(root)], check=True)
    if args.node:
        subprocess.run([args.node, str(HERE / "workbook_stage1.mjs"), str(root)], check=True)
    print(f"Stage 1 analysis finished: {root}", flush=True)


if __name__ == "__main__":
    main()
