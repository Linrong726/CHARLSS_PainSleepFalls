"""Rebuild the locked CHARLS B/W1 cohort and export aggregate evidence only."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from openpyxl import load_workbook

FROZEN = Path(__file__).resolve().parent / "frozen"
legacy_path = str(FROZEN / "legacy")
sys.path.insert(0, legacy_path)
try:
    from audit_core import GROUPS as STATE_GROUPS, exact_index, join, outcome_status
    from frozen.legacy.cohort import build_cohort
    from frozen.no_fit_validation import (
        ANALYSIS, CORE, EXTRA, add_frozen_covariates, derive_masks, make_design,
        matrix_digest, same_columns,
    )
    from frozen.tracking_patch import link_tracking, tracking_synthetic_tests
finally:
    sys.path.remove(legacy_path)
from frozen.selection_audit import (
    LABELS, audit_selection, categorical_group, continuous_group, selection_synthetic_tests,
)
from frozen.run_nonfit_v21 import (
    GROUPS, MODELS, base_counts, end_counts, month_difference, month_outputs, read_json,
    require, severity_from_2011, verify_source_values, write_csv,
)

SHEETS = ("核心变量", "连接与时间质控", "来源索引")
TIME_FIELDS = ("calendar_month_gap", "calendar_month_gap_c")
S3_REASONS = (
    "FALL_ITEM_BLANK_OR_UNDIFFERENTIATED", "FALL_DONT_KNOW", "FALL_REFUSAL",
    "PROXY_STATUS_UNCERTAIN", "PROXY_OUTCOME",
)
S3_DEFINITION = (
    "Denominator n+m: observed outcomes plus classifiable participants with actual "
    "2018 demographic and health interviews, valid return routing, exact same-person "
    "2015 anchor and strictly ordered closed months, with unavailable response or "
    "invalid proxy status. Excludes deaths and records without a verified interview/anchor."
)
BASELINE_LEVELS = {
    "sex": (1, 2), "education_R": (0, 1, 2), "rural": (0, 1),
    "prior_fall": (0, 1), "sleep_severity_2011": (0, 1), "walk100": (0, 1),
    "vision": (1, 2, 3, 4, 5), "memory": (1, 2, 3, 4, 5),
}
BASELINE_CONT = {"age": "years", "mood9": "points (0–27)", "chronic12": "conditions (0–12)"}


def dictionary_audit(base, destination):
    workbook = base / "04_VARIABLE_DICTIONARY_v2.1.1.xlsx"
    sources = base / "dictionary_csv"
    wb = load_workbook(workbook, read_only=True, data_only=True)
    checks, locations = [], {}
    try:
        require(tuple(wb.sheetnames) == SHEETS, "Dictionary sheets differ from v2.1.1")
        expected_files = {f"{idx:02}_{sheet}.csv" for idx, sheet in enumerate(SHEETS, 1)}
        require({f.name for f in sources.glob("*.csv")} == expected_files, "Dictionary CSV coverage incomplete")
        for idx, sheet in enumerate(SHEETS, 1):
            source = sources / f"{idx:02}_{sheet}.csv"
            with source.open(encoding="utf-8-sig", newline="") as handle:
                csv_rows = list(csv.reader(handle))
            xlsx_rows = [list("" if c is None else str(c) for c in row)
                         for row in wb[sheet].values]
            require(xlsx_rows == csv_rows, f"Dictionary workbook and CSV differ in {sheet}")
            for row in xlsx_rows[1:]:
                if row[1] in TIME_FIELDS:
                    locations.setdefault(row[1], []).append(sheet)
            checks.append({"sheet": sheet, "csv": source.name,
                           "rows_including_header": len(xlsx_rows), "columns": len(xlsx_rows[0]),
                           "workbook_csv_exact": True})
    finally:
        wb.close()
    require(locations == {field: [SHEETS[1]] for field in TIME_FIELDS},
            "A time field is absent or duplicated in the dictionary")
    destination.mkdir()
    shutil.copyfile(workbook, destination / workbook.name)
    for row in checks:
        shutil.copyfile(sources / row["csv"], destination / row["csv"])
    return checks


def s3_eligibility(frame):
    return (
        frame.B_class & frame.Y.isna() & ~frame.dead2.eq(1)
        & ~frame.iwstat2.isin([5, 6])
        & frame.in_demo1.eq(True) & frame.in_demo2.eq(True) & frame.in_health2.eq(True)
        & frame.route2.eq(1) & frame.new_fall2.isna()
        & frame.exact_t1_anchor & frame.t0_before_t1 & frame.t1_before_t2
        & frame.outcome_status.isin(S3_REASONS)
    )


def s3_bounds(frame):
    eligible = s3_eligibility(frame)
    rows = []
    for group in GROUPS:
        group_mask = frame.B_class & frame.Bgroup.eq(group)
        observed = frame.loc[group_mask & frame.Y.notna(), "Y"]
        n, e, m = len(observed), int(observed.eq(1).sum()), int((group_mask & eligible).sum())
        total = n + m
        rows.append({"Bgroup": group, "n": n, "e": e, "m": m, "denominator": total,
                     "lower": e / total if total else None,
                     "upper": (e + m) / total if total else None})
    return rows, eligible


def baseline_table(frame, masks):
    result = []
    for sample in ("B_class", "B_M1_R"):
        for group in ("ALL", *GROUPS):
            selection = masks[sample] if group == "ALL" else masks[sample] & frame.Bgroup.eq(group)
            q = frame.loc[selection]
            common = {"sample": sample, "Bgroup": group, "group_denominator": len(q)}
            for field, unit in BASELINE_CONT.items():
                result.append({**common, "field": field, "label": LABELS[field], "unit": unit,
                               "statistic_type": "continuous", "level": None,
                               **continuous_group(q[field])})
            for field, levels in BASELINE_LEVELS.items():
                for level in levels:
                    result.append({**common, "field": field,
                                   "label": LABELS.get(field, "基线睡眠不安频率等级"),
                                   "unit": "people / valid denominator",
                                   "statistic_type": "category_level", "level": level,
                                   **categorical_group(q[field], levels, level)})
    return result


def verify_members_and_designs(before, frame, masks, archived121, archived21, config):
    before_masks = derive_masks(before)
    require(before.index.equals(frame.index) and frame.index.is_unique
            and frame.index.str.fullmatch(r"[0-9]{12}").all(), "Canonical IDs invalid after relink")
    require(before_masks.equals(masks), "2015 relink altered the locked memberships")
    fields = same_columns(before.loc[masks.B_start, ANALYSIS], frame.loc[masks.B_start, ANALYSIS])
    require(all(r["different_values"] == r["different_missing_positions"] == 0
                and r["dtype_unchanged"] for r in fields), "Relink altered a locked field")
    for record in archived121["field_checks"]:
        require(record["different_values"] == record["different_missing_positions"] == 0
                and record["dtype_unchanged"], "Archived row-wise field check is inconsistent")
    members = []
    for record in archived121["member_checks"]:
        stage = record["stage"]
        left, right = before.index[before_masks[stage]], frame.index[masks[stage]]
        require(left.equals(right) and len(right) == record["after"]
                and record["same_ordered_member_index"] and record["flag_differences"] == 0,
                f"Ordered membership check failed for {stage}")
        members.append({"stage": stage, "rows": len(right), "ordered_keys_equal": True,
                        "member_digest_sha256": hashlib.sha256("\n".join(right).encode()).hexdigest()})
    designs = []
    for proof in archived121["matrix_checks"]:
        name = proof["model"]
        fields_model = CORE if name == "B_M1_R" else CORE + EXTRA
        q, y, x = make_design(frame, masks[name], fields_model, config["model"]["formulas"][name])
        old_q, old_y, old_x = make_design(before, before_masks[name], fields_model, proof["formula"])
        require(q.equals(old_q) and y.equals(old_y) and x.equals(old_x),
                f"Keyed archived reconstruction differs for {name}")
        require(x.columns.tolist() == proof["columns"] and matrix_digest(x) == proof["after_design_sha256"],
                f"Frozen v1.2.1 column/value hash differs for {name}")
        designs.append({"archive": "v1.2.1", "model": name, "rows": len(q),
                        "matrix_sha256": matrix_digest(x), "archive_hash_matched": True})
    for name, (formula, sample, add, nparams) in MODELS.items():
        q_src = frame.loc[masks[sample]].copy()
        for field in add:
            q_src[field] = q_src[field].astype("int64")
        fields_model = CORE + (EXTRA if sample == "B_M2_R" else []) + list(add)
        q, y, x = make_design(q_src, masks.loc[q_src.index, sample], fields_model, formula)
        proof = archived21["models"][name]
        require(q.index.equals(frame.index[masks[sample]]) and q.Y.equals(frame.loc[q.index, "Y"])
                and q.cluster.equals(frame.loc[q.index, "cluster"])
                and np.array_equal(y.to_numpy().ravel(), q.Y.to_numpy()),
                f"Model candidate indices/outcomes/communities differ for {name}")
        require(x.shape[1] == nparams and x.columns.tolist() == proof["columns"]
                and matrix_digest(x) == proof["matrix_sha256"],
                f"Frozen v2.1 column/value hash differs for {name}")
        designs.append({"archive": "v2.1", "model": name, "rows": len(q),
                        "matrix_sha256": matrix_digest(x), "archive_hash_matched": True})
    return members, fields, designs


def build(raw: Path, reference: Path, out: Path) -> pd.DataFrame:
    """Return an in-memory frame indexed by canonical string ID. Never export individual rows."""
    config = read_json(FROZEN / "legacy" / "config_v1_2.json")
    before, evidence = build_cohort(raw)
    before = add_frozen_covariates(before, config)
    frame, tracking = link_tracking(before, evidence["inputs"])
    masks = derive_masks(frame, frame.t1_confirmed_death | frame.t1_source_conflict)
    require(frame[["B_start", "B_class"]].equals(masks[["B_start", "B_class"]])
            and tracking["B_start"]["mortality_source_conflicts"] == 0,
            "Relink changed exposure eligibility or indicates a source conflict")
    observed = masks.B_class & frame.Y.notna()
    require(observed.equals(masks.B_M1_R) and not (masks.B_M2_R & ~masks.B_M1_R).any()
            and frame.loc[observed, "M2_complete"].equals(masks.loc[observed, "B_M2_R"]),
            "Complete-case masks differ from frozen samples")
    invalid = frame.dead2.eq(1) | frame.iwstat2.isin([5, 6]) & ~frame.in_demo2.eq(True)
    require(frame.loc[invalid, "Y"].isna().all(), "Confirmed death has nonmissing Y")
    frame["sleep_severity_2011"] = severity_from_2011(
        frame.sleep_code0, masks.B_start, frame.proxy0, frame.proxy_unknown0)
    frame["calendar_month_gap"] = month_difference(frame.date_lo1.where(observed), frame.date_lo2.where(observed))
    frame["calendar_month_gap_c"] = frame.calendar_month_gap - 36
    require(frame.sleep_severity_2011.notna().equals(masks.B_start)
            and frame.calendar_month_gap.notna().equals(observed)
            and frame.loc[observed, "calendar_month_gap"].gt(0).all(), "Severity or recall window outside scope")
    obs = frame.loc[observed]
    require(obs[["in_demo1", "in_demo2", "in_health2", "t0_before_t1", "t1_before_t2",
                 "exact_t1_anchor"]].eq(True).all().all()
            and obs.route2.eq(1).all() and obs.new_fall2.isna().all()
            and not obs.proxy2.eq(True).any() and not obs.proxy_unknown2.eq(True).any()
            and obs.Y.isin([0, 1]).all(), "Valid-Y record violates frozen route/anchor/proxy rules")
    lineage = verify_source_values(evidence["inputs"], frame, masks, obs)
    a121 = read_json(reference / "v121" / "01_patch" / "analysis_invariance.json")
    a21 = read_json(reference / "v21" / "08_QA_AND_REPRODUCIBILITY" / "nonfit_summary.json")
    member_checks, field_checks, design_checks = verify_members_and_designs(before, frame, masks, a121, a21, config)

    for col in masks:
        frame[col] = masks[col]
    tables, selection = audit_selection(frame)
    groups = [r | {"event_fraction": r["events"] / r["y_observed"],
                    "N_M2": int((masks.B_M2_R & frame.Bgroup.eq(r["Bgroup"])).sum())}
              for r in tables["selection_outcome_by_group"]]
    sleep_flow, sleep_by_group = base_counts(frame, masks, observed)
    months, month_frequencies, _, time_comparison = month_outputs(obs)
    # Export elapsed month frequencies only; omit calendar-month pairs that may form singleton cells.
    month_frequencies = [r for r in month_frequencies if r["field"] == "calendar_month_gap"]
    s3, s3_mask = s3_bounds(frame)
    counts = {"age60": int(frame.age.ge(60).sum()), **{k: int(masks[k].sum()) for k in masks},
              "Y_events_M1": int(obs.Y.eq(1).sum()),
              "Y_events_M2": int(frame.loc[masks.B_M2_R, "Y"].eq(1).sum())}
    require([counts[k] for k in ("age60", "B_start", "B_class", "B_M1_R", "B_M2_R",
                                  "Y_events_M1", "Y_events_M2")] == [7290, 1219, 927, 771, 595, 264, 188],
            "Archived sample and event counts differ")
    require([(r["y_observed"], r["events"], r["non_events"]) for r in groups]
            == [(341, 131, 210), (147, 55, 92), (152, 49, 103), (131, 29, 102)],
            "Archived group event counts differ")
    grades = obs.sleep_severity_2011.value_counts().to_dict()
    require(grades == {0: 299, 1: 472}
            and selection["valid_y_but_t1_unclassifiable"] == 42,
            "Archived sleep level distribution or valid-Y outside classified set differs")
    tracking_status = tracking["B_start_without_t1_demographics"]["tracking_status_counts"]
    require(tracking_status.get("NONRESPONSE_VITAL_UNKNOWN") == 51,
            "Archived 2015 vital status count differs")
    require([r["m"] for r in s3] == [12, 8, 5, 8]
            and frame.loc[s3_mask, "calendar_month_gap"].isna().all(),
            "S3 eligible-missing count or calendar field scope differs")
    reasons = (frame.loc[s3_mask].groupby(["Bgroup", "outcome_status"], observed=True)
               .size().rename("n").reset_index().to_dict("records"))

    out.mkdir(exist_ok=True)
    checks = dictionary_audit(reference / "v211", out / "dictionary")
    flow = [{"stage": key, "N": counts[key], "events": None, "non_events": None}
            for key in ("age60", "B_start", "B_class")]
    for key in ("B_M1_R", "B_M2_R"):
        q = frame.loc[masks[key]]
        flow.append({"stage": key, **{k: v for k, v in end_counts(q).items()
                                     if k in ("N", "events", "non_events")}})
    tables.update(flow=flow, groups=groups, baseline=baseline_table(frame, masks), sleep_flow=sleep_flow,
                  sleep_by_group=sleep_by_group, month_summary=months, month_frequencies=month_frequencies,
                  time_comparison=[time_comparison], s3=s3, s3_reason_breakdown=reasons,
                  cohort_member_checks=member_checks, cohort_field_checks=field_checks,
                  cohort_historical_design_hashes=design_checks, cohort_linkage_qc=evidence["inputs"].qc,
                  dictionary_qa=checks)
    dest = out / "cohort_tables"
    dest.mkdir()
    for name, rows in tables.items():
        write_csv(dest / f"{name}.csv", rows)
    require(all(row["unchanged"] for row in evidence["inputs"].verify()),
            "Raw source changed during cohort reconstruction")
    summary = dict(counts=counts, groups=groups, time=months, time_comparison=time_comparison, s3=s3,
                   s3_definition=S3_DEFINITION, selection=selection, lineage=lineage, tracking=tracking,
                   dictionary_checks=checks, ordered_membership_verified=True, historical_design_checks=design_checks,
                   source_scope={"baseline_sleep_severity_0": 299, "baseline_sleep_severity_1": 472,
                                 "vital_status_unknown_2015": 51, "unclassifiable_with_valid_y": 42})
    (out / "cohort_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
                                             + "\n", encoding="utf-8")
    return frame


def self_test():
    """Check rule edges on manufactured records. Does not instantiate or fit a model."""
    tests = []

    def check(name, passed, evidence):
        require(passed, f"SYNTHETIC {name} failed")
        tests.append({"phase": "SYNTHETIC", "test": name, "status": "PASS", "evidence": evidence})

    raw = pd.Series(["3", "4", "1", "2", ".", ".r", "3", "4", "3"])
    eligible = pd.Series([True]*8 + [False])
    full_proxy = pd.Series([False]*6 + [True, False, False])
    unknown_proxy = pd.Series([False]*7 + [True, False])
    levels = severity_from_2011(raw, eligible, full_proxy, unknown_proxy)
    check("SLEEP_2011_LEVELS", levels.iloc[:2].tolist() == [0, 1] and levels.iloc[2:].isna().all(),
          "Only DC015 3/4 in eligible self/assisted records maps to levels 0/1.")
    start = pd.to_datetime(pd.Series(["2015-12-01", "2017-01-01", "2018-02-01", None]))
    finish = pd.to_datetime(pd.Series(["2018-01-01", "2017-01-01", "2018-01-01", "2018-01-01"]))
    gaps = month_difference(start, finish)
    check("CROSS_YEAR_CALENDAR_MONTHS", gaps.iloc[:3].tolist() == [25, 0, -1] and pd.isna(gaps.iloc[3]),
          "Uses year/month arithmetic and keeps missing months missing.")
    check("GROUP_POLARITY", STATE_GROUPS == {"11": "both_persist", "10": "sleep_only",
                                            "01": "pain_only", "00": "both_resolve"},
          "2015 positive pain / negative sleep is sleep-only resolution, and vice versa.")

    q = pd.DataFrame({
        "B_class": [True] * 9, "Bgroup": ["both_persist"] * 9,
        "dead2": [0, 0, 1, 0, 0, 0, 0, 0, 0], "iwstat2": [1, 1, 5, 1, 1, 1, 9, 1, 1],
        "in_demo1": [True] * 9, "in_demo2": [True] * 6 + [False, True, True],
        "in_health2": [True] * 7 + [False, True],
        "route2": [1] * 4 + [2] + [1] * 4, "new_fall2": [np.nan] * 9,
        "fall2": [1., 0., 0., 0., 0., 0., np.nan, np.nan, np.nan],
        "fall_code2": ["1", "2", "2", "2", "2", "2", ".", ".", ".r"],
        "proxy2": [False] * 3 + [True] + [False] * 5, "proxy_unknown2": [False] * 9,
        "exact_t1_anchor": [True] * 5 + [False] + [True] * 3,
        "t0_before_t1": [True] * 9, "t1_before_t2": [True] * 9,
        "calendar_month_gap": [36, 36] + [None] * 7,
    })
    q["date_hi0"] = pd.Timestamp("2011-12-31")
    q["date_lo1"] = pd.Timestamp("2015-11-01")
    q["date_hi1"] = pd.Timestamp("2015-11-30")
    q["date_lo2"] = pd.Timestamp("2018-11-01")
    q["last_lo2"] = q.date_lo1.where(q.exact_t1_anchor)
    q["last_hi2"] = q.date_hi1.where(q.exact_t1_anchor)
    q["outcome_status"], q["Y"] = outcome_status(q, True)
    check("DEATH_INVALID_Y_STAYS_MISSING", q.Y.iloc[:2].tolist() == [1, 0] and q.Y.iloc[2:].isna().all(),
          "Death, proxy, wrong route, missing anchor, missing interview/module and refusal cannot become Y=0.")
    bounds, mask = s3_bounds(q)
    check("S3_ANCHORED_MISSING_AND_BOUNDS",
          mask[mask].index.tolist() == [3, 8] and bounds[0]["lower"] == .25 and bounds[0]["upper"] == .75
          and all(row["denominator"] == 0 and row["lower"] is None for row in bounds[1:]),
          "Anchored proxy/refusal enter m without gap values; structural exclusions stay outside n+m.")

    qc = []
    left = exact_index(pd.DataFrame({"key": ["0001", "0002"]}), "key", "synthetic", qc)
    right = exact_index(pd.DataFrame({"key": ["0002", "0001"], "x": [2, 1]}), "key", "synthetic", qc)
    linked = join(left, right, "synthetic one-to-one", qc)
    duplicate_rejected = False
    try:
        join(left, pd.concat([right, right.iloc[[0]]]), "synthetic duplicate", qc)
    except pd.errors.MergeError:
        duplicate_rejected = True
    check("EXACT_ID_LINK", linked.index.tolist() == ["0001", "0002"] and linked.x.tolist() == [1, 2]
          and duplicate_rejected, "String IDs keep leading zeroes, exact matches only, duplicate join keys fail.")
    tests.extend(tracking_synthetic_tests())
    tests.extend(selection_synthetic_tests())
    require(all(row["status"] == "PASS" for row in tests), "A reused SYNTHETIC cohort check failed")
    return tests
