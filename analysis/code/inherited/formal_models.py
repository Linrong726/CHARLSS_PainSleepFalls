"""Run and audit the five fixed modified-Poisson configurations."""

from datetime import datetime, timezone, timedelta
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from patsy import dmatrices
from scipy.linalg import solve
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf

from historical_model_audit import (
    BLOCKING_WARNING_CATEGORIES, FINITE_STATES, default_model_config,
    poisson_finite_mle, scaled_design, _iteration_changes,
)

GROUPS = ("both_persist", "sleep_only", "pain_only", "both_resolve")
LEVELS = {
    "Bgroup": GROUPS, "sex": (1., 2.), "education_R": (0., 1., 2.),
    "rural": (0., 1.), "prior_fall": (0., 1.), "sleep_severity_2011": (0, 1),
}
CORE = ["Bgroup", "sex", "education_R", "rural", "prior_fall", "sleep_severity_2011", "age_c"]
EXTRA = ["mood9", "chronic12", "walk100", "vision", "memory"]
PRIMARY = "B_M1_R_v21"
F = ('Y ~ C(Bgroup, Treatment(reference="both_persist")) + age_c + C(sex)'
     ' + C(education_R) + C(rural) + C(prior_fall) + C(sleep_severity_2011)')
MODELS = {
    PRIMARY: (F, "main", 11, "sole_primary", CORE),
    "B_S1_NOPRIOR_v21": (F.replace(" + C(prior_fall)", ""), "main", 10, "S1",
                         [field for field in CORE if field != "prior_fall"]),
    "B_S2_BRIDGE_v21": (F, "extended", 11, "S2_bridge", CORE),
    "B_M2_R_v21": (F + " + " + " + ".join(EXTRA), "extended", 16, "S2_extended", CORE + EXTRA),
    "B_TIME_R_v21": (F + " + calendar_month_gap_c", "main", 12, "S4", CORE + ["calendar_month_gap_c"]),
}
THRESHOLDS = default_model_config()["thresholds"]
ATTEMPTS = default_model_config()["attempts"]
ACCEPTANCE = default_model_config()["acceptance"]
COV_TOL = {"atol": 1e-8, "rtol": 1e-6}
REPRO_TOL = {"atol": 1e-8, "rtol": 1e-7}
Z975 = 1.959963984540054


def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="microseconds")


def frozen_config():
    return {
        "models": {name: dict(formula=f, sample=s, columns=p, role=r, fields=v)
                   for name, (f, s, p, r, v) in MODELS.items()},
        "levels_and_references": {k: {"levels": list(v), "reference": v[0]}
                                  for k, v in LEVELS.items()},
        "sample_rules": {
            "main": "frozen B/W1 classifiable, valid Y, complete core including prior_fall and community",
            "extended": "same core eligibility plus five frozen extended fields complete",
            "s3": "classifiable and missing Y due to proxy or response only, known interview and "
                  "return route with verified anchor and order, exclude death",
        },
        "fit": {"family": "Poisson", "link": "log", "method": "IRLS", "scale": 1,
                "tol_criterion": "deviance", "rtol": 0, "atol_equals_tol": True,
                "cluster": "2011 communityID", "cov_type": "cluster", "use_correction": True,
                "df_correction": True, "use_t": False, "z975": Z975,
                "weight": None, "offset": None, "exposure": None, "attempts": ATTEMPTS},
        "diagnostics": THRESHOLDS, "dual_fit_tolerance": ACCEPTANCE,
        "sandwich_tolerance": COV_TOL, "reproduction_tolerance": REPRO_TOL,
        "reproduction_excludes": ["clock timestamps", "runtime paths", "run phase and log metadata"],
    }


def matrix_digest(matrix):
    h = hashlib.sha256("\n".join(matrix.columns).encode())
    h.update(np.ascontiguousarray(matrix.to_numpy(), dtype="<f8").tobytes())
    return h.hexdigest()


def model_input(data, name):
    formula, _, p, _, fields = MODELS[name]
    if not data.index.is_unique:
        raise ValueError("Input must have one row per participant")
    q = data[["Y", "cluster", *fields]].copy()
    if q.isna().any().any() or set(q.Y) != {0, 1}:
        raise ValueError(f"Invalid complete-case mask or binary outcome for {name}")
    for field, levels in LEVELS.items():
        if field not in q:
            continue
        if set(q[field]) != set(levels):
            raise ValueError(f"Frozen category levels differ: {name}, {field}")
        q[field] = pd.Categorical(q[field], categories=levels)
    # Keep the frozen archive's int-valued time category and numeric category labels.
    for field in ["Y", *[v for v in fields if v not in LEVELS]]:
        q[field] = q[field].astype(float)
    y, x = dmatrices(formula, q, NA_action="raise", return_type="dataframe")
    if not (x.index.equals(q.index) and y.index.equals(q.index) and x.shape[1] == p
            and np.array_equal(y.to_numpy().ravel(), q.Y.to_numpy()) and np.isfinite(x).all().all()):
        raise ValueError(f"Unexpected design columns, values or membership: {name}")
    for term, levels in LEVELS.items():
        if term not in q:
            continue
        prefix = "C(Bgroup," if term == "Bgroup" else f"C({term})"
        actual = [col.split("[T.")[-1][:-1] for col in x if col.startswith(prefix)]
        if actual != list(map(str, levels[1:])):
            raise ValueError(f"Unexpected treatment columns/reference: {name}, {term}, {actual}")
    return q, y, x


def comparison(a, b, limits):
    return {
        "max_abs_difference": float(np.max(np.abs(a - b))),
        "n_exceeding_tolerance": int((~np.isclose(a, b, **limits)).sum()),
        **limits,
    }


def sandwich(x, y, mu, groups):
    """Independently sum cluster scores and solve against positive information."""
    n, p = x.shape
    g = int(groups.max()) + 1
    info = x.T @ (mu[:, None] * x)
    scores = np.zeros((g, p))
    np.add.at(scores, groups, x * (y - mu)[:, None])
    transformed = solve(info, scores.T, assume_a="pos")
    factor = g / (g - 1) * (n - 1) / (n - p)
    return factor * transformed @ transformed.T, factor


def estimate_arrays(fit):
    return dict(coefficient=np.asarray(fit.params), robust_se=np.asarray(fit.bse),
                covariance=np.asarray(fit.cov_params()), prediction=np.asarray(fit.predict()))


def ci(beta, se):
    return np.exp(beta), np.exp(beta - Z975 * se), np.exp(beta + Z975 * se)


def log_event(log, **record):
    with log.open("a", encoding="utf-8") as file:
        file.write(json.dumps({"timestamp": timestamp(), **record}, ensure_ascii=False) + "\n")


def fit_once(name, phase, q, y, x, z, groups, attempt, log):
    info = {"model": name, **attempt, "n": len(q), "fit_invoked": False, "warnings": [], "blockers": []}
    blockers = info["blockers"]
    fit = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            formula = MODELS[name][0]
            model = smf.glm(formula, q, family=sm.families.Poisson(link=sm.families.links.Log()), missing="raise")
            # Check the exact outcome, design, ordered participants and group vector supplied to fit.
            if (list(model.exog_names) != x.columns.tolist()
                    or not np.array_equal(model.exog, x.to_numpy())
                    or not np.array_equal(model.endog, y)
                    or not model.data.row_labels.equals(q.index)
                    or not np.array_equal(groups, pd.factorize(q.cluster, sort=False)[0])):
                raise ValueError("Model input differs from locked row order, X, Y or clusters")
            log_event(log, event="fit_invoked", phase=phase, model=name, attempt=attempt["name"],
                      n=len(q), tol=attempt["tol"], maxiter=attempt["maxiter"])
            info["fit_invoked"] = True
            fit = model.fit(
                method="IRLS", tol=attempt["tol"], atol=attempt["tol"], rtol=0.,
                tol_criterion="deviance", maxiter=attempt["maxiter"], scale=1.,
                cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True,
                                              "df_correction": True}, use_t=False,
            )
            if (fit.nobs != len(q) or not fit.model.data.row_labels.equals(q.index)
                    or not np.array_equal(fit.model.endog, y)
                    or not np.array_equal(fit.model.exog, x.to_numpy())
                    or not np.array_equal(fit.cov_kwds["groups"], groups)
                    or fit.cov_type != "cluster" or fit.use_t or fit.scale != 1.
                    or not fit.cov_kwds["use_correction"] or not fit.cov_kwds["adjust_df"]):
                raise ValueError("Returned fit does not preserve the frozen sample or covariance settings")
            info["postfit_input_verified"] = True
        except (ValueError, RuntimeError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            blockers.append(f"{type(exc).__name__}: {exc}")
    for w in caught:
        category = w.category.__name__
        msg = f"{category}: {w.message}"
        (blockers if category in BLOCKING_WARNING_CATEGORIES else info["warnings"]).append(msg)
    arrays = None
    if fit is not None and not blockers:
        arrays = estimate_arrays(fit)
        beta, se, cov, mu = (arrays[k] for k in ("coefficient", "robust_se", "covariance", "prediction"))
        if not fit.converged:
            blockers.append("IRLS did not converge")
        if not all(np.isfinite(a).all() for a in (*arrays.values(), *ci(beta, se))) or not np.isfinite(fit.deviance):
            blockers.append("Nonfinite estimates, covariance, intervals, means or deviance")
        elif (se <= 0).any() or (mu <= 0).any():
            blockers.append("Nonpositive robust SE or fitted means")
        else:
            eig_min = np.linalg.eigvalsh((cov + cov.T) / 2).min()
            cov_rank = np.linalg.matrix_rank(cov)
            weighted = z * np.sqrt(mu)[:, None]
            condition = float(np.linalg.cond(weighted))
            score = float(np.max(np.abs(z.T @ (y - mu))) / len(y))
            info.update(score_scaled=score, weighted_condition=condition, covariance_rank=int(cov_rank),
                        covariance_min_eigenvalue=float(eig_min),
                        max_abs_beta=float(np.max(np.abs(beta))), prediction_gt1=int((mu > 1).sum()),
                        prediction_max=float(mu.max()))
            if cov_rank != x.shape[1] or eig_min < -THRESHOLDS["covariance_psd_rel_tol"] * np.max(np.abs(cov)):
                blockers.append("Invalid cluster covariance rank or eigenvalue")
            if np.linalg.matrix_rank(weighted) != x.shape[1] or condition > THRESHOLDS["max_scaled_condition"]:
                blockers.append("Weighted design rank/condition failed")
            if not np.isfinite(score) or score > THRESHOLDS["score_scaled_inf_per_n"]:
                blockers.append("Scaled score exceeds threshold")
            if info["max_abs_beta"] > THRESHOLDS["max_abs_coef_warning"]:
                info["warnings"].append("Extreme log-scale coefficient")
        change = _iteration_changes(fit, x.shape[1])
        info.update(converged=bool(fit.converged), iterations=int(fit.fit_history["iteration"]),
                    last_step_max_abs=float(np.max(np.abs(change))) if np.isfinite(change).all() else None)
    log_event(log, event="fit_diagnosed", phase=phase, **info)
    return info, arrays, fit


def fit_model(name, phase, q, y_frame, x_frame, log, tables):
    _, _, p, role, fields = MODELS[name]
    y, x = y_frame.to_numpy().ravel(), x_frame.to_numpy()
    z = scaled_design(x)
    groups, _ = pd.factorize(q.cluster, sort=False)
    sizes = np.bincount(groups)
    n, g, events = len(q), len(sizes), int(y.sum())
    rank, event_rank = int(np.linalg.matrix_rank(z)), int(np.linalg.matrix_rank(z[y == 1]))
    condition = float(np.linalg.cond(z))
    label = {"model": name}
    result = {**label, "status": "NOT_RUN", "n": n, "events": events, "clusters": g, "p": p,
              "rank": rank, "event_rank": event_rank, "nonevent_rank": int(np.linalg.matrix_rank(z[y == 0])),
              "scaled_condition": condition if np.isfinite(condition) else None, "finite_mle": None,
              "score_scaled": None, "prediction_gt1": None, "prediction_gt1_pct": None,
              "prediction_max": None, "warnings": "", "blockers": ""}
    tables["design_summary"].append({
        **result, "formula": MODELS[name][0], "matrix_sha256": matrix_digest(x_frame),
        "event_clusters": int(np.unique(groups[y == 1]).size),
        "nonevent_clusters": int(np.unique(groups[y == 0]).size),
        "cluster_size_min": int(sizes.min()), "cluster_size_max": int(sizes.max()),
        "cluster_size_median": float(np.median(sizes)), "events_per_parameter": events / p,
    })
    design = tables["design_summary"][-1]
    del design["status"], design["score_scaled"], design["prediction_gt1"], design["prediction_gt1_pct"]
    del design["prediction_max"], design["warnings"], design["blockers"], design["finite_mle"]
    if name == "B_TIME_R_v21":
        j = x_frame.columns.get_loc("calendar_month_gap_c")
        residual = z[:, j] - np.delete(z, j, axis=1) @ np.linalg.lstsq(
            np.delete(z, j, axis=1), z[:, j], rcond=None)[0]
        design["time_vif"] = float(np.dot(z[:, j], z[:, j]) / np.dot(residual, residual))
    cautions, blockers = [], []
    sparse = False
    supported = {"walk100": (0, 1), "vision": (1, 2, 3, 4, 5), "memory": (1, 2, 3, 4, 5)}
    for field in fields:
        levels = LEVELS.get(field, supported.get(field, ()))
        for level in levels:
            selected = q[field].eq(level)
            count = int(selected.sum())
            case = int(q.loc[selected, "Y"].sum())
            sparse |= min(case, count - case) < THRESHOLDS["small_level_outcomes_warning"]
            for group in ("ALL", *GROUPS):
                part = q.loc[selected if group == "ALL" else selected & q.Bgroup.eq(group)]
                tables["model_levels"].append({**label, "variable": field, "level": level,
                                               "group": group, "n": len(part), "events": int(part.Y.sum()),
                                               "clusters": int(part.cluster.nunique())})
    for field in [v for v in fields if v not in LEVELS and v not in supported]:
        for group in GROUPS:
            part = q.loc[q.Bgroup.eq(group), field]
            tables["model_score_coverage"].append({**label, "group": group, "variable": field,
                                                   "n": len(part), "mean": float(part.mean()),
                                                   "sd": float(part.std()), "min": float(part.min()),
                                                   "max": float(part.max())})
    if min(int(q.loc[q.Bgroup.eq(group), "Y"].sum()) for group in (GROUPS[0], GROUPS[-1])) < THRESHOLDS["primary_events_advisory"]:
        cautions.append("Primary contrast arm has fewer than 30 events")
    if min(events, n - events) / p < THRESHOLDS["preferred_events_per_parameter"]:
        cautions.append("Fewer than 15 events or nonevents per parameter")
    if g < THRESHOLDS["small_cluster_warning"]:
        cautions.append("Fewer than 30 communities")
    if sparse:
        cautions.append("Sparse observed covariate levels (some scored linearly)")
    if np.linalg.matrix_rank(x) != p or rank != p or g <= p or n <= p:
        blockers.append("Rank or cluster coverage insufficient")
    elif not np.isfinite(condition) or condition > THRESHOLDS["max_scaled_condition"]:
        blockers.append("Scaled design condition exceeds frozen threshold")
    else:
        existence, _ = poisson_finite_mle(z, y, THRESHOLDS)
        result["finite_mle"] = existence["status"]
        tables["finite_solution"].append({**label, **existence})
        if existence["status"] not in FINITE_STATES:
            blockers.append(existence["status"])
    if blockers:
        return result | {"status": "NOT_ESTIMABLE", "warnings": " | ".join(cautions),
                         "blockers": " | ".join(blockers)}

    first, strict = [fit_once(name, phase, q, y, x_frame, z, groups, a, log) for a in ATTEMPTS]
    for details, _, _ in (first, strict):
        tables["fit_attempts"].append({k: (" | ".join(v) if isinstance(v, list) else v)
                                      for k, v in details.items()})
        blockers.extend(f"{details['name']}: {msg}" for msg in details["blockers"])
        cautions.extend(f"{details['name']}: {msg}" for msg in details["warnings"])
    if first[2] is None or strict[2] is None or first[2].model is strict[2].model:
        blockers.append("Two independent fitted objects could not be verified")
    if blockers:
        return result | {"status": "FAIL", "warnings": " | ".join(cautions),
                         "blockers": " | ".join(blockers)}
    a, b = first[1], strict[1]
    for metric in ACCEPTANCE:
        check = comparison(a[metric], b[metric], ACCEPTANCE[metric])
        tables["numerical_checks"].append({**label, "test": "dual_IRLS", "metric": metric, **check})
        if check["n_exceeding_tolerance"]:
            blockers.append(f"IRLS attempts disagree for {metric}")
    beta, se, cov, mu = (b[k] for k in ("coefficient", "robust_se", "covariance", "prediction"))
    try:
        independent, correction = sandwich(x, y, mu, groups)
        alt_se = np.sqrt(np.diag(independent))
        for metric, candidate, reference in (
            ("covariance", independent, cov), ("robust_se", alt_se, se),
            ("rr_ci_low", ci(beta, alt_se)[1], ci(beta, se)[1]),
            ("rr_ci_high", ci(beta, alt_se)[2], ci(beta, se)[2]),
        ):
            check = comparison(candidate, reference, COV_TOL)
            tables["numerical_checks"].append({**label, "test": "independent_sandwich", "metric": metric,
                                               "correction": correction, **check})
            if check["n_exceeding_tolerance"]:
                blockers.append(f"Independent sandwich check differs for {metric}")
    except (np.linalg.LinAlgError, ValueError) as exc:
        blockers.append(f"Independent sandwich failed: {exc}")
    for group in ("ALL", *GROUPS):
        selected = np.ones(n, dtype=bool) if group == "ALL" else q.Bgroup.eq(group).to_numpy()
        fitted = mu[selected]
        tables["prediction_summary"].append({**label, "group": group, "n": len(fitted),
                                             "gt1": int((fitted > 1).sum()),
                                             "gt1_pct": float(100 * np.mean(fitted > 1)),
                                             "prediction_max": float(fitted.max())})
    gt1 = int((mu > 1).sum())
    if gt1:
        cautions.append("Fitted means above one, retained without clipping")
    status = "FAIL" if blockers else "PASS_WITH_WARNINGS" if cautions else "PASS"
    result.update(status=status, score_scaled=strict[0]["score_scaled"],
                  prediction_gt1=gt1, prediction_gt1_pct=100 * gt1 / n,
                  prediction_max=float(mu.max()), warnings=" | ".join(cautions),
                  blockers=" | ".join(blockers))
    if blockers:
        return result
    rr, lo, hi = ci(beta, se)
    for i, term in enumerate(x_frame.columns):
        stats = {**label, "term": term, "n": n, "events": events, "clusters": g, "beta": beta[i],
                 "se": se[i], "rr": rr[i], "ci_low": lo[i], "ci_high": hi[i],
                 "p_value": 2 * norm.sf(abs(beta[i] / se[i]))}
        tables["coefficients"].append(stats)
        for j, other in enumerate(x_frame.columns):
            tables["covariance"].append({**label, "term_1": term, "term_2": other, "covariance": cov[i, j]})
        if term.startswith("C(Bgroup,"):
            group = term.split("[T.")[-1][:-1]
            contrast_role = "secondary_description" if name == PRIMARY and group != GROUPS[-1] else role
            tables["contrasts"].append({**stats, "role": contrast_role, "comparison": group, "status": status})
    return result


def prepare(main, extended, prior_design=None):
    if not extended.index.isin(main.index).all() or not main.loc[extended.index, ["Y", "cluster", *CORE]].equals(
            extended[["Y", "cluster", *CORE]]):
        raise ValueError("Extended sample is not the identical locked subset of main inputs")
    samples = {"main": main, "extended": extended}
    designs = {name: model_input(samples[mask], name) for name, (_, mask, *_) in MODELS.items()}
    if prior_design is not None:
        for name in (PRIMARY, "B_M2_R_v21", "B_TIME_R_v21"):
            archived = prior_design.loc[prior_design.model.eq(name)].iloc[0]
            design = designs[name][2]
            if matrix_digest(design) != archived.matrix_sha256:
                raise ValueError(f"Historical v2.1 design digest differs for {name}")
    return designs


def run(designs, phase, log):
    tables = {k: [] for k in ("contrasts", "coefficients", "covariance", "diagnostics", "design_summary",
                             "model_levels", "model_score_coverage", "finite_solution", "fit_attempts",
                             "numerical_checks", "prediction_summary")}
    for name in MODELS:
        result = fit_model(name, phase, *designs[name], log, tables)
        tables["diagnostics"].append(result)
        print(f"{phase} {name} {result['status']} N={result['n']}", flush=True)
    return tables
