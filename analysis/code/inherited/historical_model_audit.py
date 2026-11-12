"""Fixed B/W1 modified Poisson technical diagnostics. Emit aggregates only."""

from __future__ import annotations

from collections import Counter
import logging
import warnings

import numpy as np
import pandas as pd
from patsy import dmatrices
from scipy.optimize import linprog
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf


LOG = logging.getLogger(__name__)
GROUPS = ("both_persist", "sleep_only", "pain_only", "both_resolve")
M1 = "B_M1_R"
M2 = "B_M2_R"
BASE = (
    'Y ~ C(Bgroup, Treatment(reference="both_persist")) + age_c + C(sex)'
    ' + C(education_R) + C(rural) + C(prior_fall)'
)
EXTRA = " + mood9 + chronic12 + walk100 + vision + memory"
FORMULAS = {M1: BASE, M2: BASE + EXTRA}
CATEGORICAL_LEVELS = {
    "Bgroup": GROUPS,
    "sex": (1, 2),
    "education_R": (0, 1, 2),
    "rural": (0, 1),
    "prior_fall": (0, 1),
    "walk100": (0, 1),
    "vision": (1, 2, 3, 4, 5),
    "memory": (1, 2, 3, 4, 5),
}
CORE_LEVELS = ("Bgroup", "sex", "education_R", "rural", "prior_fall")
EXTRA_LEVELS = ("walk100", "vision", "memory")
CORE_SCORES = ("age_c",)
EXTRA_SCORES = ("mood9", "chronic12")
BLOCKING_WARNING_CATEGORIES = {
    "ConvergenceWarning", "PerfectSeparationWarning", "HessianInversionWarning",
    "IterationLimitWarning", "SingularMatrixWarning", "RuntimeWarning",
}
FINITE_STATES = {"FINITE_POSITIVE_ROWS_FULL_RANK", "FINITE_NO_RECESSION_DIRECTION"}
TABLE_NAMES = (
    "model_inputs", "model_category_levels", "model_exposure_covariates",
    "model_cluster_sizes", "model_fits", "model_coefficients", "model_covariance",
    "model_numeric_comparisons", "model_predictions", "model_prediction_comparisons",
    "model_warnings", "model_primary_contrasts", "model_separation_direction",
)


def default_model_config() -> dict:
    """Project rules to freeze before the first fit to the actual B data."""
    return {
        "formulas": FORMULAS.copy(),
        "attempts": [
            {"name": "first", "tol": 1e-8, "maxiter": 500},
            {"name": "strict", "tol": 1e-10, "maxiter": 1000},
        ],
        "covariance": {"use_correction": True, "df_correction": True,
                       "use_t": False, "alpha": 0.05},
        "thresholds": {
            "max_scaled_condition": 1e8,
            "score_scaled_inf_per_n": 1e-7,
            "lp_feasibility": 1e-9,
            "lp_gain": 1e-8,
            "max_abs_coef_warning": 10.0,
            "small_level_outcomes_warning": 5,
            "preferred_events_per_parameter": 15,
            "primary_events_advisory": 30,
            "small_cluster_warning": 30,
            "covariance_psd_rel_tol": 1e-9,
        },
        "acceptance": {
            "coefficient": {"atol": 1e-6, "rtol": 1e-6},
            "robust_se": {"atol": 1e-6, "rtol": 1e-6},
            "covariance": {"atol": 1e-6, "rtol": 1e-6},
            "prediction": {"atol": 1e-7, "rtol": 1e-6},
        },
    }


def scaled_design(x: np.ndarray) -> np.ndarray:
    """Center and unit-SD scale nonconstant columns; retain the intercept."""
    sd = x.std(axis=0, ddof=0)
    active = sd > 0
    z = x.copy()
    z[:, active] = (x[:, active] - x[:, active].mean(axis=0)) / sd[active]
    return z


def _max_abs(v: np.ndarray) -> float:
    return float(np.max(np.abs(v))) if v.size else 0.0


def _finite_number(value) -> float | None:
    return float(value) if np.isfinite(value) else None


def poisson_finite_mle(z: np.ndarray, y: np.ndarray, thresholds: dict) -> tuple[dict, np.ndarray | None]:
    """Existence test for a full-column-rank Poisson log-link design.

    The recession condition is X+ d = 0, X0 d <= 0, and X0 d < 0 for
    at least one zero-outcome row. Positive-row full rank excludes it.
    Otherwise minimize mean(X0 d) under bounded d using HiGHS. The cube
    contains a scaled copy of every direction. Numerical ambiguity blocks
    technical acceptance rather than being treated as finite existence.
    See Correia, Guimaraes and Zylkin, https://arxiv.org/abs/1903.01633.
    """
    p = z.shape[1]
    pos, zero = z[y > 0], z[y == 0]
    pos_rank = int(np.linalg.matrix_rank(pos)) if pos.size else 0
    result = {
        "positive_rows": int(len(pos)), "zero_rows": int(len(zero)),
        "positive_rank": pos_rank, "p": p, "solver_status": None,
        "solver_message": None, "objective_gain_mean": None,
        "max_positive_equality_abs": None, "max_zero_inequality_violation": None,
        "min_zero_margin": None, "n_strict_negative_zero_rows": None,
    }
    if pos_rank == p:
        return result | {"status": "FINITE_POSITIVE_ROWS_FULL_RANK"}, None

    feasibility = thresholds["lp_feasibility"]
    gain_tol = thresholds["lp_gain"]
    lp = linprog(
        zero.mean(axis=0), A_ub=zero, b_ub=np.zeros(len(zero)),
        A_eq=pos, b_eq=np.zeros(len(pos)), bounds=[(-1.0, 1.0)] * p,
        method="highs", options={"primal_feasibility_tolerance": feasibility,
                                 "dual_feasibility_tolerance": feasibility},
    )
    result.update(solver_status=int(lp.status), solver_message=str(lp.message))
    if not lp.success or lp.x is None:
        return result | {"status": "INDETERMINATE_LP_FAILURE"}, None

    pos_margin, zero_margin = pos @ lp.x, zero @ lp.x
    equality_error = _max_abs(pos_margin)
    inequality_error = float(np.maximum(0.0, zero_margin.max()))
    gain = float(-zero_margin.mean())
    result.update(objective_gain_mean=gain, max_positive_equality_abs=equality_error,
                  max_zero_inequality_violation=inequality_error,
                  min_zero_margin=float(zero_margin.min()),
                  n_strict_negative_zero_rows=int(np.sum(zero_margin < -gain_tol)))
    if equality_error > 10 * feasibility or inequality_error > 10 * feasibility:
        return result | {"status": "INDETERMINATE_LP_RESIDUAL"}, lp.x
    if gain > gain_tol and zero_margin.min() < -gain_tol:
        return result | {"status": "NO_FINITE_MLE_RECESSION_DIRECTION"}, lp.x
    if abs(gain) > feasibility or zero_margin.min() < -gain_tol:
        return result | {"status": "INDETERMINATE_NEAR_LP_TOLERANCE"}, lp.x
    return result | {"status": "FINITE_NO_RECESSION_DIRECTION"}, None


def _counts(d: pd.DataFrame) -> dict:
    return {"n": len(d), "events": int(d.Y.eq(1).sum()),
            "non_events": int(d.Y.eq(0).sum()), "clusters": int(d.cluster.nunique())}


def _distribution(v: np.ndarray) -> dict:
    if not len(v):
        return dict.fromkeys(("min", "p05", "p25", "median", "p75", "p95", "max", "mean", "sd"))
    q = np.quantile(v, [0, .05, .25, .5, .75, .95, 1])
    result = dict(zip(("min", "p05", "p25", "median", "p75", "p95", "max"), map(_finite_number, q)))
    return result | {"mean": _finite_number(v.mean()),
                     "sd": _finite_number(v.std(ddof=1)) if len(v) > 1 else None}


def _categorical_tables(d, fields, scores, label, tables, threshold):
    for field in fields:
        for level in CATEGORICAL_LEVELS[field]:
            part = d.loc[d[field].eq(level)]
            counts = _counts(part)
            low = min(counts["events"], counts["non_events"])
            tables["model_category_levels"].append(label | dict(
                variable=field, level=level, **counts,
                support="EMPTY" if not counts["n"] else (
                    "ZERO_OUTCOME_CELL" if low == 0 else ("SPARSE" if low < threshold else "OBSERVED")),
            ))
            if field == "Bgroup":
                continue
            for group in GROUPS:
                cross = part.loc[part.Bgroup.eq(group)]
                tables["model_exposure_covariates"].append(label | dict(
                    Bgroup=group, variable=field, level=level, variable_kind="categorical_or_ordinal",
                    **_counts(cross),
                ))
    for group in GROUPS:
        part = d.loc[d.Bgroup.eq(group)]
        for field in scores:
            tables["model_exposure_covariates"].append(label | dict(
                Bgroup=group, variable=field, variable_kind="numeric", **_counts(part),
                **_distribution(part[field].to_numpy(dtype=float)),
            ))


def _within(a, b, tolerance: dict) -> np.ndarray:
    return np.isclose(a, b, atol=tolerance["atol"], rtol=tolerance["rtol"], equal_nan=False)


def _compare_attempts(a, b, limits, label, tables, terms, group_values):
    result = {}
    for metric in ("coefficient", "robust_se", "covariance", "prediction"):
        first, strict = a[metric], b[metric]
        delta = strict - first
        within = _within(first, strict, limits[metric])
        result[metric] = dict(max_abs_difference=_finite_number(_max_abs(delta)),
                              n_exceeding_tolerance=int((~within).sum()), **limits[metric])
        if metric in ("coefficient", "robust_se"):
            for j, term in enumerate(terms):
                tables["model_numeric_comparisons"].append(label | dict(
                    metric=metric, term=term, first=_finite_number(first[j]),
                    strict=_finite_number(strict[j]), signed_difference=_finite_number(delta[j]),
                    abs_difference=_finite_number(abs(delta[j])),
                    within_tolerance=bool(within[j]), **limits[metric],
                ))
        if metric == "covariance":
            for i, term_i in enumerate(terms):
                for j, term_j in enumerate(terms):
                    tables["model_covariance"].append(label | dict(
                        term_i=term_i, term_j=term_j, first=_finite_number(first[i, j]),
                        strict=_finite_number(strict[i, j]), signed_difference=_finite_number(delta[i, j]),
                        abs_difference=_finite_number(abs(delta[i, j])),
                        within_tolerance=bool(within[i, j]), **limits[metric],
                    ))
        if metric == "prediction":
            for group in ("ALL", *GROUPS):
                selected = np.ones(len(first), dtype=bool) if group == "ALL" else group_values == group
                absolute = np.abs(delta[selected])
                tables["model_prediction_comparisons"].append(label | dict(
                    Bgroup=group, n=int(selected.sum()), n_exceeding_tolerance=int((~within[selected]).sum()),
                    mean_signed_difference=_finite_number(delta[selected].mean()),
                    **{f"abs_difference_{k}": v for k, v in _distribution(absolute).items()},
                    **limits[metric],
                ))
    result["status"] = "PASS" if all(v["n_exceeding_tolerance"] == 0 for v in result.values()) else "FAIL"
    return result


def _iteration_changes(fit, p: int) -> np.ndarray:
    history = [np.asarray(v, dtype=float) for v in fit.fit_history.get("params", [])
               if np.shape(v) == (p,)]
    if len(history) < 2:
        return np.full(p, np.nan)
    return history[-1] - history[-2]


def _fit_once(d, formula, x, z, y, terms, attempt, cov_cfg, thresholds, label, tables):
    attempt_label = label | {"attempt": attempt["name"]}
    result = dict(**attempt_label, formula=formula, tol=attempt["tol"], maxiter=attempt["maxiter"],
                  irls_tolerance_criterion="deviance", irls_rtol=0.0)
    blocking = []
    cov_kwds = {"groups": pd.factorize(d.cluster, sort=False)[0],
                "use_correction": cov_cfg["use_correction"], "df_correction": cov_cfg["df_correction"]}
    model = fit = numeric = None
    fit_invoked = False
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = smf.glm(formula=formula, data=d, family=sm.families.Poisson(link=sm.families.links.Log()),
                            missing="raise")
            if list(model.exog_names) != terms or not np.array_equal(model.exog, x) or not np.array_equal(model.endog, y):
                raise ValueError("GLM design differs from the frozen pre-fit design")
            LOG.info("%s %s %s fit called with N=%d tol=%.1g maxiter=%d", label["phase"], label["model"],
                     attempt["name"], len(d), attempt["tol"], attempt["maxiter"])
            fit_invoked = True
            fit = model.fit(method="IRLS", tol=attempt["tol"], maxiter=attempt["maxiter"],
                            tol_criterion="deviance", rtol=0.0, cov_type="cluster",
                            cov_kwds=cov_kwds, use_t=cov_cfg["use_t"])
            coef = np.asarray(fit.params, dtype=float)
            se = np.asarray(fit.bse, dtype=float)
            cov = np.asarray(fit.cov_params(), dtype=float)
            pred = np.asarray(fit.predict(), dtype=float)
            critical = norm.ppf(1 - cov_cfg["alpha"] / 2)
            ci_lo, ci_hi = coef - critical * se, coef + critical * se
            rr, rr_lo, rr_hi = np.exp(coef), np.exp(ci_lo), np.exp(ci_hi)
            numeric = dict(coefficient=coef, robust_se=se, covariance=cov, prediction=pred)
    except (ValueError, RuntimeError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        blocking.append(f"{type(exc).__name__}: {exc}")

    captured = []
    for entry in caught:
        category, message = entry.category.__name__, str(entry.message)
        fatal = category in BLOCKING_WARNING_CATEGORIES
        captured.append(dict(category=category, message=message, blocking=fatal))
        if fatal:
            blocking.append(f"{category}: {message}")
        tables["model_warnings"].append(attempt_label | captured[-1])

    result.update(fit_invoked=fit_invoked, fit_returned=fit is not None)
    if numeric is None:
        return result | dict(status="FAIL", converged=None, blocking_reasons=blocking,
                             captured_warnings=captured), None, model

    assert fit is not None
    if not fit.converged:
        blocking.append("IRLS did not converge")
    arrays = (coef, se, cov, pred, rr, rr_lo, rr_hi)
    if not all(np.isfinite(a).all() for a in arrays) or not np.isfinite(fit.deviance):
        blocking.append("nonfinite estimates, standard errors, covariance, intervals, means or deviance")
    if (se <= 0).any() or (pred <= 0).any() or any((r <= 0).any() for r in (rr, rr_lo, rr_hi)):
        blocking.append("nonpositive standard errors, means or exponentiated estimates/intervals")
    eig_min = None
    cov_rank = None
    if np.isfinite(cov).all():
        eig_min = float(np.linalg.eigvalsh((cov + cov.T) / 2).min())
        cov_rank = int(np.linalg.matrix_rank(cov))
        if eig_min < -thresholds["covariance_psd_rel_tol"] * max(_max_abs(cov), np.finfo(float).tiny):
            blocking.append("cluster covariance fails positive semidefiniteness check")
        if cov_rank != len(terms):
            blocking.append("cluster covariance is rank-deficient")
    score_inf = _max_abs(z.T @ (y - pred)) / len(d)
    if not np.isfinite(score_inf) or score_inf > thresholds["score_scaled_inf_per_n"]:
        blocking.append("scaled score residual exceeds threshold")
    weighted_cond = None
    weighted_rank = None
    if np.isfinite(pred).all() and (pred > 0).all():
        weighted = z * np.sqrt(pred)[:, None]
        weighted_rank = int(np.linalg.matrix_rank(weighted))
        weighted_cond = _finite_number(np.linalg.cond(weighted))
        if weighted_rank < len(terms) or weighted_cond is None or weighted_cond > thresholds["max_scaled_condition"]:
            blocking.append("weighted scaled design ill-conditioned or rank-deficient")

    changes = _iteration_changes(fit, len(terms))
    caution = []
    if _max_abs(coef) > thresholds["max_abs_coef_warning"]:
        caution.append("extreme log-scale coefficient; inspect iteration history and finite-solution check")
    pred_gt1 = int((pred > 1).sum())
    if pred_gt1:
        caution.append("fitted means above one; out-of-range probability warning, values not clipped")
    for message in caution:
        tables["model_warnings"].append(attempt_label | dict(category="DiagnosticWarning", message=message, blocking=False))

    for i, term in enumerate(terms):
        row = attempt_label | dict(term=term, coefficient=_finite_number(coef[i]),
                                    robust_se=_finite_number(se[i]), RR=_finite_number(rr[i]),
                                    CI95_low=_finite_number(rr_lo[i]), CI95_high=_finite_number(rr_hi[i]),
                                    log_CI95_low=_finite_number(ci_lo[i]), log_CI95_high=_finite_number(ci_hi[i]),
                                    last_irls_step=_finite_number(changes[i]),
                                    extreme_coef_warning=bool(abs(coef[i]) > thresholds["max_abs_coef_warning"]))
        tables["model_coefficients"].append(row)
        if term.startswith("C(Bgroup,"):
            group = term.rsplit("[T.", 1)[1][:-1]
            tables["model_primary_contrasts"].append(row | dict(
                comparison=group + " vs both_persist", role="PRIMARY" if group == "both_resolve" else "SECONDARY",
            ))
    for group in ("ALL", *GROUPS):
        values = pred if group == "ALL" else pred[d.Bgroup.eq(group).to_numpy()]
        count = int((values > 1).sum())
        tables["model_predictions"].append(attempt_label | dict(
            Bgroup=group, n=len(values), n_above_one=count,
            proportion_above_one=count / len(values), **_distribution(values),
        ))

    result.update(status="FAIL" if blocking else "PASS", converged=bool(fit.converged),
                  iterations=int(fit.fit_history["iteration"]), deviance=_finite_number(fit.deviance),
                  log_likelihood=_finite_number(fit.llf), scale=_finite_number(fit.scale),
                  score_scaled_inf_per_n=_finite_number(score_inf),
                  max_abs_coefficient=_finite_number(_max_abs(coef)),
                  max_abs_final_irls_step=_finite_number(_max_abs(changes)),
                  min_cov_eigenvalue=eig_min, covariance_rank=cov_rank,
                  weighted_scaled_rank=weighted_rank, weighted_scaled_condition=weighted_cond,
                  weighted_information_condition=weighted_cond**2 if weighted_cond is not None else None,
                  prediction_n_above_one=pred_gt1, prediction_proportion_above_one=pred_gt1 / len(d),
                  prediction_max=_finite_number(pred.max()), blocking_reasons=blocking,
                  captured_warnings=captured, diagnostic_warnings=caution)
    return result, numeric, model


def fit_model(d: pd.DataFrame, name: str, cfg: dict, *, phase: str = "REAL") -> tuple[dict, dict]:
    """Fit only the fixed revised B model, with two independent GLM objects."""
    formula = cfg["formulas"][name]
    if formula != FORMULAS[name]:
        raise ValueError(f"Unexpected frozen formula for {name}")
    if cfg["covariance"]["use_t"] or cfg["covariance"]["alpha"] != 0.05:
        raise ValueError("Model audit requires normal two-sided 95% Wald intervals")
    if cfg["attempts"] != default_model_config()["attempts"]:
        raise ValueError("Expected the frozen pair of independent IRLS attempts")
    thresholds = cfg["thresholds"]
    tables = {key: [] for key in TABLE_NAMES}
    label = {"model": name, "phase": phase}
    blocking, cautions = [], []
    summary = label | dict(formula=formula, status="NOT_RUN", prefit={}, attempts=[], comparison=None,
                           blocking_reasons=blocking, warnings=cautions, n_fit_attempts=0,
                           distinct_fit_objects_verified=False)
    fields = CORE_LEVELS + (EXTRA_LEVELS if name == M2 else ())
    scores = CORE_SCORES + (EXTRA_SCORES if name == M2 else ())
    keep = ["Y", "cluster", *fields, *scores]
    if not d.index.is_unique:
        blocking.append("sample index is not unique")
    if not len(d) or not d[keep].notna().all().all():
        blocking.append("no complete input or unexpected missing values; no rows dropped by model")
    if not d.Y.isin((0, 1)).all() or d.Y.nunique() != 2:
        blocking.append("outcome does not contain both valid binary values")
    for field in fields:
        if not d[field].isin(CATEGORICAL_LEVELS[field]).all():
            blocking.append(f"unexpected categorical value in {field}")
    if set(d.Bgroup) != set(GROUPS):
        blocking.append("a required B exposure group is absent")
    if not np.isfinite(d[list(scores)].to_numpy(dtype=float)).all():
        blocking.append("nonfinite numeric covariate")
    if blocking:
        return tables, summary

    d = d[keep].copy()
    d["Bgroup"] = pd.Categorical(d.Bgroup, categories=GROUPS)
    _categorical_tables(d, fields, scores, label, tables, thresholds["small_level_outcomes_warning"])
    y_frame, x_frame = dmatrices(formula, d, NA_action="raise", return_type="dataframe")
    y, x, terms = y_frame.to_numpy(dtype=float).ravel(), x_frame.to_numpy(dtype=float), list(x_frame.columns)
    assert x_frame.index.equals(d.index) and np.array_equal(y, d.Y.to_numpy(dtype=float))
    z = scaled_design(x)
    rank, scaled_rank = int(np.linalg.matrix_rank(x)), int(np.linalg.matrix_rank(z))
    p, cond = x.shape[1], _finite_number(np.linalg.cond(z))
    sizes = d.cluster.value_counts(sort=False).to_numpy(dtype=int)
    for size, count in sorted(Counter(sizes.tolist()).items()):
        tables["model_cluster_sizes"].append(label | dict(size=size, n_clusters=count, n_participants=size * count))
    counts = _counts(d)
    budget = min(counts["events"], counts["non_events"]) / p
    groups = {group: _counts(d.loc[d.Bgroup.eq(group)]) for group in GROUPS}
    cov_cfg = cfg["covariance"]
    adjustment = None
    n, g = len(d), counts["clusters"]
    if n > p and g > 1:
        adjustment = g / (g - 1) * (n - 1) / (n - p) if cov_cfg["use_correction"] else 1.0
    prefit = counts | dict(p=p, rank=rank, scaled_rank=scaled_rank, scaled_condition=cond,
                           scaled_condition_method="center and unit population-SD scale nonconstant columns",
                           min_outcome_per_parameter=budget, cluster_key="t0 raw communityID",
                           cluster_size_min=int(sizes.min()), cluster_size_median=float(np.median(sizes)),
                           cluster_size_max=int(sizes.max()), covariance="one-way cluster sandwich",
                           covariance_small_sample_factor=adjustment,
                           **cov_cfg, ci_method="exp(beta ± 1.959963984540054 × cluster robust SE)",
                           finite_mle_status=None)
    summary["prefit"] = prefit | dict(groups=groups)
    if rank != p or scaled_rank != p or cond is None or cond > thresholds["max_scaled_condition"]:
        blocking.append("pre-fit design ill-conditioned or rank-deficient")
    if g <= p or n <= p:
        blocking.append("insufficient independent clusters or observations for cluster covariance")
    if not blocking:
        existence, direction = poisson_finite_mle(z, y, thresholds)
        prefit["finite_mle_status"] = existence["status"]
        summary["prefit"].update(finite_mle_status=existence["status"], finite_mle=existence)
        if direction is not None:
            for term, weight in zip(terms, direction):
                tables["model_separation_direction"].append(label | dict(term=term, scaled_design_weight=float(weight)))
        if existence["status"] not in FINITE_STATES:
            blocking.append(existence["status"])
    tables["model_inputs"].append(label | prefit)
    if blocking:
        return tables, summary

    if min(groups["both_persist"]["events"], groups["both_resolve"]["events"]) < thresholds["primary_events_advisory"]:
        cautions.append("primary arm has fewer than 30 observed events (advisory only)")
    if budget < thresholds["preferred_events_per_parameter"]:
        cautions.append("outcome count per parameter below preferred 15 (advisory only)")
    if g < thresholds["small_cluster_warning"]:
        cautions.append("small number of clusters (advisory only)")
    if any(row["support"] in ("ZERO_OUTCOME_CELL", "SPARSE") for row in tables["model_category_levels"]):
        cautions.append("sparse covariate level(s); see category distribution and finite-solution test")
    for message in cautions:
        tables["model_warnings"].append(label | dict(attempt="prefit", category="Advisory", message=message, blocking=False))

    estimates, models = [], []
    for attempt in cfg["attempts"]:
        info, numeric, model = _fit_once(d, formula, x, z, y, terms, attempt, cov_cfg, thresholds, label, tables)
        summary["attempts"].append(info)
        summary["n_fit_attempts"] += int(info["fit_invoked"])
        estimates.append(numeric)
        models.append(model)
        for reason in info["blocking_reasons"]:
            blocking.append(f"{attempt['name']}: {reason}")
        for diagnostic in info.get("diagnostic_warnings", []):
            cautions.append(f"{attempt['name']}: {diagnostic}")
        for warning in info["captured_warnings"]:
            if not warning["blocking"]:
                cautions.append(f"{attempt['name']}: {warning['category']}: {warning['message']}")
        tables["model_fits"].append(info | dict(
            blocking_reasons=" | ".join(info["blocking_reasons"]),
            diagnostic_warnings=" | ".join(info.get("diagnostic_warnings", [])),
            captured_warnings=" | ".join(f"{w['category']}: {w['message']}" for w in info["captured_warnings"]),
        ))

    distinct = all(model is not None for model in models) and models[0] is not models[1]
    summary["distinct_fit_objects_verified"] = distinct
    if not distinct:
        blocking.append("two separate GLM objects could not be verified")
    if all(e is not None for e in estimates):
        comparison = _compare_attempts(*estimates, cfg["acceptance"], label, tables, terms, d.Bgroup.to_numpy())
        summary["comparison"] = comparison
        if comparison["status"] != "PASS":
            blocking.append("strict-tolerance refit disagrees for at least one coefficient, SE, covariance entry or prediction")
            tables["model_warnings"].append(label | dict(
                attempt="comparison", category="NumericalDrift", message=blocking[-1], blocking=True,
            ))
    else:
        blocking.append("two-attempt numerical comparison unavailable")
    summary["status"] = "FAIL" if blocking else "PASS"
    summary["primary_strict"] = next((row for row in tables["model_primary_contrasts"]
                                      if row["attempt"] == "strict" and row["role"] == "PRIMARY"), None)
    LOG.info("%s %s status=%s N=%d events=%d attempts=%d finite_mle=%s blockers=%s warnings=%s",
             phase, name, summary["status"], n, counts["events"], summary["n_fit_attempts"],
             prefit["finite_mle_status"], blocking, cautions)
    return tables, summary


def synthetic_model_qa(cfg: dict) -> list[dict]:
    """Check the same numerical machinery on generated records only."""
    rng = np.random.default_rng(241107)
    n = 400
    y = np.zeros(n, dtype=int)
    for g, events in enumerate((30, 30, 30, 29)):
        y[g * 100 + rng.choice(100, events, replace=False)] = 1
    d = pd.DataFrame(dict(Y=y, Bgroup=np.repeat(GROUPS, 100), cluster=np.tile(np.arange(50), 8),
                          age_c=rng.normal(5, 6, n), sex=rng.integers(1, 3, n),
                          education_R=rng.integers(0, 3, n), rural=rng.integers(0, 2, n),
                          prior_fall=rng.integers(0, 2, n)))
    _, fit = fit_model(d, M1, cfg, phase="SYNTHETIC")
    primary = fit["primary_strict"]
    stable = (fit["status"] == "PASS" and fit["n_fit_attempts"] == 2
              and fit["distinct_fit_objects_verified"])
    spans_one = bool(primary and primary["CI95_low"] <= 1 <= primary["CI95_high"])

    # Full overall rank yet a category has no events. A naive convergence flag cannot override the LP.
    separated = d.copy()
    separated.loc[separated.education_R.eq(2), "Y"] = 0
    LOG.info("SYNTHETIC naive IRLS demonstration on separated generated sample")
    naive = smf.glm(BASE, data=separated, family=sm.families.Poisson()).fit(tol=1e-8, maxiter=500)
    _, guard = fit_model(separated, M1, cfg, phase="SYNTHETIC")
    no_mle = guard["prefit"].get("finite_mle_status") == "NO_FINITE_MLE_RECESSION_DIRECTION"
    blocks_false_convergence = (bool(naive.converged) and no_mle and guard["n_fit_attempts"] == 0
                                and guard["status"] == "NOT_RUN")
    # Positive rows alone have rank 1, but both signs occur among the zeros.
    lp_overlap, _ = poisson_finite_mle(
        scaled_design(np.array([[1.0, 0.0], [1.0, 1.0], [1.0, -1.0]])),
        np.array([1, 0, 0]), cfg["thresholds"],
    )

    # Two mock converged estimates differ materially for a nuisance coefficient.
    arrays_a = dict(coefficient=np.array([0.0, 0.0]), robust_se=np.ones(2),
                    covariance=np.eye(2), prediction=np.full(4, 0.2))
    arrays_b = arrays_a | {"coefficient": np.array([0.0, 0.05])}
    comparison_tables = {key: [] for key in TABLE_NAMES}
    comparison = _compare_attempts(arrays_a, arrays_b, cfg["acceptance"],
                                   {"model": M1, "phase": "SYNTHETIC"}, comparison_tables,
                                   ["exposure", "nuisance"], np.asarray(GROUPS))
    flags_drift = comparison["status"] == "FAIL" and comparison["coefficient"]["n_exceeding_tolerance"] == 1
    return [
        {"phase": "SYNTHETIC", "test": "SYNTHETIC_29_EVENTS_AND_CI_CONTAINS_ONE",
         "status": "PASS" if stable and spans_one else "FAIL",
         "detail": f"29 main-arm events, two actual synthetic fits, status={fit['status']}, CI contains 1={spans_one}"},
        {"phase": "SYNTHETIC", "test": "SYNTHETIC_POISSON_ZERO_EVENT_LEVEL_GUARD",
         "status": "PASS" if blocks_false_convergence else "FAIL",
         "detail": f"Naive IRLS converged={bool(naive.converged)}; audited status={guard['prefit'].get('finite_mle_status')}, attempts={guard['n_fit_attempts']}, LP objective gain={guard['prefit'].get('finite_mle', {}).get('objective_gain_mean')}"},
        {"phase": "SYNTHETIC", "test": "SYNTHETIC_POISSON_LP_FINITE_WITH_RANK_DEFICIENT_POSITIVE_ROWS",
         "status": "PASS" if lp_overlap["status"] == "FINITE_NO_RECESSION_DIRECTION" else "FAIL",
         "detail": f"Positive-row rank={lp_overlap['positive_rank']} of 2, LP status={lp_overlap['status']}"},
        {"phase": "SYNTHETIC", "test": "SYNTHETIC_REFIT_NUISANCE_DRIFT_IS_RECORDED",
         "status": "PASS" if flags_drift else "FAIL",
         "detail": f"Mock converged fits with nuisance change=0.05, comparison={comparison['status']}; {comparison['coefficient']}"},
    ]
