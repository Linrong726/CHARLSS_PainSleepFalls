"""Read-only ANALYSIS-1.1 source audit. Export aggregates and digests only."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import re
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import pandas as pd
from pandas.io.stata import StataMissingValue, StataReader


ZIP_SHA = "d8aef12514a14e8ed1454fa710f68317237a2458d4cb4bcc718367d12243063a"
GROUPS = ("both_persist", "sleep_only", "pain_only", "both_resolve")
DISEASES = (
    ("hypertension", "r1hibpe", "r1hibpf", "da007_1_"),
    ("dyslipidemia", "r1dyslipe", "r1dyslipf", "da007_2_"),
    ("diabetes_high_blood_sugar", "r1diabe", "r1diabf", "da007_3_"),
    ("cancer", "r1cancre", "r1cancrf", "da007_4_"),
    ("chronic_lung_disease", "r1lunge", "r1lungf", "da007_5_"),
    ("liver_disease", "r1livere", "r1liverf", "da007_6_"),
    ("heart_disease", "r1hearte", "r1heartf", "da007_7_"),
    ("stroke", "r1stroke", "r1strokf", "da007_8_"),
    ("kidney_disease", "r1kidneye", "r1kidneyf", "da007_9_"),
    ("digestive_disease", "r1digeste", "r1digestf", "da007_10_"),
    ("arthritis_rheumatism", "r1arthre", "r1arthrf", "da007_13_"),
    ("asthma", "r1asthmae", "r1asthmaf", "da007_14_"),
)
WALK_ITEMS = ("db001", "db002", "db003")
DO_ZIP = Path("Harmonized CHARLS/H_CHARLS_D_do_file.zip")
DOC = Path("2011 CHARLS Wave1/CHARLS2011________.doc")
CODEBOOK = Path("Harmonized CHARLS/Harmonized_CHARLS_D.pdf")
RAW_HEALTH = Path("2011 CHARLS Wave1/household_and_community_questionnaire_data/health_status_and_functioning.dta")
HARMONIZED = Path("Harmonized CHARLS/H_CHARLS_D_Data.zip")
WALK_LEVELS = {
    "1": "No difficulty", "2": "Difficulty but can do it",
    "3": "Difficulty and needs help", "4": "Cannot do it",
}


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def sha(path):
    with path.open("rb") as fp:
        return hashlib.file_digest(fp, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    table = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    table.to_csv(path, index=False, encoding="utf-8-sig")


def iso_time():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def safe_member(name):
    p = PurePosixPath(name)
    return (p.as_posix() == name and not p.is_absolute() and ".." not in p.parts
            and not set("\\:") & set(name))


def verify_baseline(zip_path, baseline, crosscheck=None):
    """Verify and use the one exact prior release. Never overwrite an existing file."""
    require(sha(zip_path) == ZIP_SHA, "Baseline ZIP hash differs from pinned ANALYSIS-1.0")
    checked = {}
    with ZipFile(zip_path) as z:
        require(z.testzip() is None, "Baseline ZIP CRC check failed")
        manifest = json.loads(z.read("FILE_MANIFEST.json"))
        sums = {}
        for line in z.read("SHA256SUMS.txt").decode("utf-8-sig").splitlines():
            digest, name = line.split(maxsplit=1)
            require(re.fullmatch(r"[0-9a-f]{64}", digest) and safe_member(name) and name not in sums,
                    "Malformed baseline sums")
            sums[name] = digest
        zip_names = z.namelist()
        require(len(zip_names) == len(set(zip_names)) and all(safe_member(n) for n in zip_names)
                and set(zip_names) == set(sums) | {"SHA256SUMS.txt"}
                and set(sums) == set(manifest["files"]) | {"FILE_MANIFEST.json"},
                "Baseline member coverage disagrees")
        for name in zip_names:
            payload = z.read(name)
            actual = hashlib.sha256(payload).hexdigest()
            if name != "SHA256SUMS.txt":
                require(actual == sums[name], f"Baseline member digest mismatch: {name}")
            if name in manifest["files"]:
                require(manifest["files"][name] == {"bytes": len(payload), "sha256": actual},
                        f"Baseline file manifest mismatch: {name}")
            target = baseline / name
            require(target.resolve().is_relative_to(baseline.resolve()), "Unsafe baseline extraction target")
            if target.exists():
                require(sha(target) == actual, f"Prior staged baseline digest differs: {name}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                require(sha(target) == actual, f"Baseline extraction digest differs: {name}")
            checked[name] = actual
    required = {"ANALYSIS_REPORT.docx", "RESULTS.xlsx", "CHAT_REVIEW.md", "ANALYSIS_MANIFEST.json",
                "FROZEN_CONFIG.json", "MEMBER_LOCK.json", "ENTRY_CHECK.json", "REPRODUCTION.json",
                "FINAL_QA.json", "code/run_analysis1.py", "code/frozen/no_fit_validation.py",
                "dictionary/04_VARIABLE_DICTIONARY_v2.1.1.xlsx",
                "study_reference/v21/01_PROTOCOL_v2.1_CANDIDATE.docx",
                "study_reference/v21/02_SAP_v2.1_CANDIDATE.docx",
                "study_reference/v21/06_ANALYSIS_SPECIFICATION_v2.1/analysis_spec_v2.1.json"}
    require(required <= set(checked), "A required baseline artifact is missing")
    run_identity = {}
    if crosscheck and crosscheck.is_dir():
        for name in ("ANALYSIS_MANIFEST.json", "FROZEN_CONFIG.json", "MEMBER_LOCK.json", "ENTRY_CHECK.json",
                     "REPRODUCTION.json", "FINAL_QA.json", "ANALYSIS_REPORT.docx", "RESULTS.xlsx"):
            file = crosscheck / name
            if file.is_file():
                run_identity[name] = sha(file) == checked[name]
        require(all(run_identity.values()), "Located 1.0 original-run artifacts disagree with the verified ZIP")
    return {"path": str(zip_path), "sha256": ZIP_SHA, "verified_zip_members": len(checked),
            "sums_sha256": checked["SHA256SUMS.txt"], "manifest_sha256": checked["FILE_MANIFEST.json"],
            "extracted_dir": str(baseline), "original_run": str(crosscheck) if crosscheck else None,
            "original_run_crosscheck": run_identity, "required_members_present": sorted(required)}


@contextmanager
def import_baseline(baseline):
    code = str((baseline / "code").resolve())
    sys.path.insert(0, code)
    try:
        import cohort
        import entry
        import formal_models as models
        import run_analysis1
        from frozen.no_fit_validation import NoFit
        from frozen.legacy import audit_core

        for m in (cohort, entry, models, run_analysis1, audit_core):
            require(Path(m.__file__).resolve().is_relative_to(Path(code)),
                    f"Wrong implementation imported for {m.__name__}")
        yield cohort, entry, models, run_analysis1, NoFit, audit_core
    finally:
        sys.path.remove(code)


def reconstruct_locked(raw, reference, baseline, temporary, impl):
    cohort, _, models, formal, _, _ = impl
    old = read_json(baseline / "ANALYSIS_MANIFEST.json")
    require(formal.env() == old["environment"], "Runtime differs from the 1.0 frozen environment")
    require(formal.code_hashes() == old["code_sha256"], "Baseline analysis code differs from freeze")
    require(sha(baseline / "FROZEN_CONFIG.json") == old["config_sha256"], "1.0 freeze config digest differs")
    frame = cohort.build(raw, reference, temporary)
    main, extended = (frame.loc[frame[col]] for col in ("B_M1_R", "B_M2_R"))
    lock = formal.row_membership(frame, main, extended)
    require(lock == read_json(baseline / "MEMBER_LOCK.json"), "Locked member/core-field digest differs from 1.0")
    compared_tables = ["cohort_member_checks.csv", "cohort_field_checks.csv", "cohort_historical_design_hashes.csv"]
    require(all(sha(temporary / "cohort_tables" / name) == sha(baseline / "cohort_tables" / name)
                for name in compared_tables), "Historical per-stage member or field evidence differs")
    prior_design = pd.read_csv(baseline / "study_reference/v21/tables/design_summary.csv")
    designs = models.prepare(main, extended, prior_design)
    archived = pd.read_csv(baseline / "model_tables/design_summary.csv").set_index("model")
    design_checks = []
    for name, (data, y, x) in designs.items():
        evidence = archived.loc[name]
        match = (models.matrix_digest(x) == evidence.matrix_sha256
                 and data.shape[0] == evidence.n and int(y.to_numpy().sum()) == evidence.events
                 and data.cluster.nunique() == evidence.clusters and x.shape[1] == evidence.p)
        require(match, f"1.0 design, cohort or events disagree: {name}")
        design_checks.append({"model": name, "n": len(data), "p": x.shape[1],
                              "events": int(y.to_numpy().sum()), "communities": int(data.cluster.nunique()),
                              "matrix_sha256": evidence.matrix_sha256, "exact_match": True})
    return frame, {"member_lock_exact_match": True, "member_lock": lock,
                   "three_member_and_field_evidence_csvs_exact": compared_tables,
                   "five_design_checks": design_checks, "same_software_environment": formal.env()}


def stata_code(value):
    if isinstance(value, StataMissingValue):
        return value.string
    if pd.isna(value):
        return "UNEXPECTED_NA"
    if isinstance(value, str):
        return value
    return str(int(value)) if value == int(value) else str(value)


def read_fields(source, fields, source_rel, file_version, mapping):
    with StataReader(source, convert_categoricals=False, convert_missing=True) as reader:
        labels, values = reader.variable_labels(), reader.value_labels()
        value_label_names = dict(zip(labels, reader._lbllist))
        lookup = {name.lower(): name for name in labels}
        desired = list(fields)
        version_field = lookup.get("versionid")
        resolved = {key: lookup[key.lower()] for key in desired if key.lower() in lookup}
        requested = list(resolved.values())
        if version_field and version_field not in requested:
            requested.append(version_field)
        frame = reader.read(columns=requested)
        release = file_version
        if version_field:
            release += " / versionID " + "|".join(sorted(frame[version_field].map(stata_code).unique()))
        metadata = []
        for key in desired:
            name = resolved.get(key)
            counts = {} if name is None or key.lower().endswith("id") or key.lower() == "id_w1" else Counter(
                frame[name].map(stata_code))
            code_labels = {} if name is None else values.get(value_label_names.get(name), {})
            metadata.append({"requested_variable": key, "actual_variable": name, "variable_label": labels.get(name),
                             "file_value_labels_json": json.dumps({str(k): v for k, v in code_labels.items()},
                                                                  ensure_ascii=False, sort_keys=True),
                             "source_missing_codes_full_release": json.dumps({k: int(v) for k, v in counts.items()
                                                                             if k.startswith(".")}, sort_keys=True),
                             "source_role": mapping.get(key, "Verified identifier used only in memory"),
                             "source_relative_path": source_rel, "release": release,
                             "stata_storage_format": reader._format_version, "source_timestamp": reader._time_stamp,
                             "status": "PRESENT" if name else f"ABSENT after case-insensitive scan of {len(labels)} fields"})
        return frame.rename(columns={v: k for k, v in resolved.items()})[list(resolved)], metadata


def codes(frame, fields):
    return pd.DataFrame({col: frame[col].map(stata_code) for col in fields}, index=frame.index)


def raw_count(raw_codes):
    """Only the twelve explicitly valid doctor-diagnosis answers form the raw count."""
    require(raw_codes.shape[1] == len(DISEASES), "Expected all twelve raw conditions")
    return raw_codes.map({"1": 1., "2": 0.}.get).sum(axis=1, min_count=len(DISEASES))


def read_audit_fields(raw, frame, core):
    disease_mapping = {raw: f"{disease}: baseline doctor-diagnosis item (1 yes, 2 no); all 12 valid for raw count"
                       for disease, _, _, raw in DISEASES}
    disease_mapping |= {harm: f"{disease}: retrospectively harmonized baseline state (0 no, 1 yes)"
                        for disease, harm, _, _ in DISEASES}
    disease_mapping |= {flag: f"{disease}: baseline dispute flag; 5/6 indicate later dispute"
                        for disease, _, flag, _ in DISEASES}
    disease_mapping |= {flag.replace("r1", "r2", 1): f"{disease}: optional first follow-up corroboration (2/3/4)"
                        for disease, _, flag, _ in DISEASES}
    mapping = disease_mapping | {
        "db001": "Run/jog 1 km; response 1 skips to DB004",
        "db002": "Walk 1 km; response 1 skips DB003",
        "db003": "Walk 100 m; response 1 no difficulty, 2/3/4 any difficulty",
        "r1walk100a": "Harmonized 0 no difficulty, 1 difficulty; retain Stata system and extended missing",
    }
    disease_raw = [raw_field for _, _, _, raw_field in DISEASES]
    disease_h = [harm for _, harm, _, _ in DISEASES]
    disease_f = [flag for _, _, flag, _ in DISEASES]
    r2_flags = [flag.replace("r1", "r2", 1) for flag in disease_f]
    raw_h, md_raw = read_fields(raw / RAW_HEALTH, ("ID", *WALK_ITEMS, *disease_raw),
                                RAW_HEALTH.as_posix(), "CHARLS 2011 Wave 1", mapping)
    with ZipFile(raw / HARMONIZED) as z, z.open("H_CHARLS_D_Data.dta") as f:
        harmon, md_h = read_fields(f, ("ID", "ID_w1", "r1walk100a", *disease_h, *disease_f, *r2_flags),
                                  HARMONIZED.as_posix() + "::H_CHARLS_D_Data.dta", "Harmonized CHARLS Version D", mapping)
    absent = [r["requested_variable"] for r in md_raw + md_h if r["status"] != "PRESENT"]
    required = {"ID", "ID_w1", "r1walk100a", *WALK_ITEMS, *disease_raw, *disease_h}
    require(not set(absent) & required, "Critical source fields absent after metadata scan: " + str(sorted(set(absent) & required)))
    metadata = md_raw + md_h
    for row in metadata:
        item = row["requested_variable"]
        label = str(row["variable_label"]).lower()
        if item in disease_f or item in r2_flags:
            require(row["status"] != "PRESENT" or ("flag" in label and "dispute" in label),
                    f"Dispute flag label mismatch: {item}")
    qc = []
    hi = core.exact_index(harmon, "ID", "Harmonized D independent baseline fields", qc)
    bridge_frame = hi.loc[hi.ID_w1.ne(""), ["ID_w1"]].reset_index()
    bridge = core.exact_index(bridge_frame, "ID_w1", "Independent official baseline bridge", qc)["ID"]
    require(bridge.index.str.fullmatch(r"[0-9]{11}").all()
            and pd.Index(bridge).str.fullmatch(r"[0-9]{12}").all() and bridge.is_unique,
            "Official bridge format or one-to-one check failed")
    hi = hi.drop(columns="ID_w1")
    ri = core.canonical(raw_h, 2011, bridge, "Wave 1 independent A01/A02 health", qc)
    main = frame.loc[frame.B_M1_R].copy()
    require(main.index.isin(hi.index).all() and main.index.isin(ri.index).all(),
            "An ordered main participant did not join to both source modules")
    hi = hi.reindex(main.index)
    ri = ri.reindex(main.index)
    raw_codes = codes(ri, [*WALK_ITEMS, *disease_raw])
    harmonized_codes = codes(hi, ["r1walk100a", *disease_h, *[c for c in disease_f + r2_flags if c in hi]])
    require(not raw_codes.eq("UNEXPECTED_NA").any().any()
            and not harmonized_codes.eq("UNEXPECTED_NA").any().any(),
            "Stata reader lost the original extended or system missing representation")
    for optional in set(absent) & set(disease_f + r2_flags):
        harmonized_codes[optional] = "FIELD_ABSENT"
    wc = harmonized_codes.r1walk100a.map({"0": 0., "1": 1.})
    require(wc.equals(main.walk100), "Frozen walk100 disagrees with independent raw read")
    hc = harmonized_codes[disease_h].map({"0": 0., "1": 1.}.get).sum(axis=1, min_count=len(DISEASES))
    require(hc.equals(main.chronic12), "Frozen chronic12 disagrees with independent raw read")
    main["chronic12_raw2011"] = raw_count(raw_codes[disease_raw])
    return main, raw_codes, harmonized_codes, metadata, qc


def walk_reason(walk, db1, db2, db3):
    if walk in {"0", "1"}:
        return "available", "direct released value"
    if db3 in WALK_LEVELS:
        return "raw_DB003_valid_but_harmonized_missing", "direct source disagreement"
    if walk == ".r" or db3 == ".r":
        return "explicit_refusal_code", "direct code"
    if walk == ".d" or db3 == ".d":
        return "explicit_dont_know_code", "direct code"
    if walk == ".a":
        return "age_inconsistent_structural_code", "code observed; standard healthy-under-50 explanation does not apply by assumption"
    if walk == ".m" and db3 == "." and "1" not in (db1, db2) and "4" in (db1, db2):
        return "consistent_with_known_wave1_routing_problem", "pattern-compatible; no person-specific routing proof"
    return "cannot_further_attribute", "no diagnostic reason in released source"


def value_counts(data, dimensions):
    return data.value_counts(dimensions, dropna=False, sort=False).rename("n").reset_index()


def write_walk(main, raw_codes, h_codes, out):
    data = main[["Bgroup", "B_M2_R", "walk100", "mood9", "chronic12", "vision", "memory"]].copy()
    data["r1walk100a_code"] = h_codes.r1walk100a
    data = data.join(raw_codes[list(WALK_ITEMS)], validate="one_to_one")
    reason = [walk_reason(*row) for row in data[["r1walk100a_code", *WALK_ITEMS]].itertuples(index=False, name=None)]
    data[["reason", "evidence_level"]] = pd.DataFrame(reason, index=data.index)
    missing = data.walk100.isna()
    codes_table = value_counts(data, ["r1walk100a_code"])
    combinations = value_counts(data.loc[missing], ["r1walk100a_code", *WALK_ITEMS, "reason", "evidence_level"])
    write_csv(out / "A01_WALK_CODE_FREQUENCIES.csv", codes_table)
    write_csv(out / "A01_WALK_MISSING_DB_COMBINATIONS.csv", combinations)
    availability = []
    for group in (*GROUPS, "ALL"):
        part = data if group == "ALL" else data.loc[data.Bgroup.eq(group)]
        availability.append({"group": group, "N": len(part), "no_difficulty": int(part.walk100.eq(0).sum()),
                             "any_difficulty": int(part.walk100.eq(1).sum()), "available": int(part.walk100.notna().sum()),
                             "missing": int(part.walk100.isna().sum()), "missing_fraction": float(part.walk100.isna().mean())})
    write_csv(out / "A01_WALK_BY_GROUP.csv", availability)
    reason_table = value_counts(data.loc[missing], ["reason", "evidence_level"])
    write_csv(out / "A01_WALK_REASON_COUNTS.csv", reason_table)
    covars = ["walk100", "mood9", "chronic12", "vision", "memory"]
    pattern = data[covars].isna().apply(lambda row: "+".join(row.index[row]) or "none", axis=1)
    write_csv(out / "A01_EXTENDED_MISSING_MUTUALLY_EXCLUSIVE.csv", value_counts(
        pd.DataFrame({"extended_retained": data.B_M2_R, "missing_covariates": pattern}),
        ["extended_retained", "missing_covariates"]))
    marginal = [{"extended_retained": bool(retain), "denominator": len(part), "covariate": c,
                 "missing": int(part[c].isna().sum())} for retain, part in data.groupby("B_M2_R") for c in covars]
    write_csv(out / "A01_EXTENDED_MISSING_MARGINAL.csv", marginal)

    raw_expected = data.db003.map({"1": "0", "2": "1", "3": "1", "4": "1", ".d": ".d", ".r": ".r"})
    normal_skip = data.db003.eq(".") & (data.db001.eq("1") | data.db002.eq("1"))
    raw_expected.loc[normal_skip] = "0"
    data["expected_from_direct_response_or_normal_skip"] = raw_expected.fillna("not_determinable")
    write_csv(out / "A01_WALK_CONSTRUCTION_CHECK.csv", value_counts(
        data, ["expected_from_direct_response_or_normal_skip", "r1walk100a_code"]))
    discordance = raw_expected.notna() & raw_expected.ne(data.r1walk100a_code)
    require(int(combinations.n.sum()) == int(missing.sum()) == int(reason_table.n.sum()),
            "Walk combinations do not reconcile to walk missing count")
    return {"main_n": len(data), "missing": int(missing.sum()), "valid": int((~missing).sum()),
            "codes": {r["r1walk100a_code"]: r["n"] for r in codes_table.to_dict("records")},
            "missing_reasons": {r["reason"]: r["n"] for r in reason_table.to_dict("records")},
            "by_group": availability,
            "documented_direct_or_normal_skip_checks": int(raw_expected.notna().sum()),
            "documented_construction_disagreements": int(discordance.sum()),
            "raw_DB003_valid_but_harmonized_missing": int((missing & data.db003.isin(WALK_LEVELS)).sum()),
            "extended_retained": int(data.B_M2_R.sum()), "extended_not_retained": int((~data.B_M2_R).sum()),
            "scope_note": "Individual CAPI paths unavailable; pattern compatibility is distinct from directly coded causes."}


def chronic_source(main, raw_codes, h_codes, out):
    long_tables, summaries, partitions, timing, totals = [], {}, [], [], []
    for name, part in (("main_771", main), ("extended_595", main.loc[main.B_M2_R])):
        changed = pd.Series(False, index=part.index)
        has_revised = changed.copy()
        item_both_valid = pd.Series(0, index=part.index)
        count_raw = part.chronic12_raw2011
        comparable = count_raw.notna() & part.chronic12.notna()
        count_different = comparable & part.chronic12.ne(count_raw)
        for disease, h, flag, raw in DISEASES:
            r2 = flag.replace("r1", "r2", 1)
            data = pd.DataFrame({
                "raw_2011_code": raw_codes.loc[part.index, raw],
                "harmonized_code": h_codes.loc[part.index, h],
                "baseline_flag_code": h_codes.loc[part.index, flag],
                "followup_wave2_flag_code": h_codes.loc[part.index, r2],
            })
            raw_status = data.raw_2011_code.map({"1": "1", "2": "0"})
            item_valid = raw_status.notna() & data.harmonized_code.isin(("0", "1"))
            item_change = item_valid & raw_status.ne(data.harmonized_code)
            revised = data.baseline_flag_code.isin(("5", "6"))
            corroborated_w2 = ((data.baseline_flag_code.eq("5") & data.followup_wave2_flag_code.eq("2"))
                               | (data.baseline_flag_code.eq("6") & data.followup_wave2_flag_code.isin(("3", "4"))))
            changed |= item_change
            has_revised |= revised
            item_both_valid += item_valid.astype(int)
            grouped = value_counts(data, list(data.columns))
            long_tables.append(grouped.assign(sample=name, disease=disease, sample_n=len(part)))
            timing.append({"sample": name, "disease": disease, "sample_n": len(part),
                           "comparable_item_n": int(item_valid.sum()),
                           "raw_status_different_n": int(item_change.sum()),
                           "status_different_with_baseline_flag_5_or_6": int((item_change & revised).sum()),
                           "baseline_flag_5_or_6": int(revised.sum()),
                           "matching_2013_dispute_flag": int(corroborated_w2.sum()),
                           "later_dispute_exact_wave_unresolved": int((revised & ~corroborated_w2).sum())})
        conditions = {
            "both_counts_complete_different": comparable & count_different,
            "both_counts_complete_same_composition_differs": comparable & ~count_different & changed,
            "both_counts_complete_same_composition_same": comparable & ~count_different & ~changed,
            "only_harmonized_count_complete": count_raw.isna() & part.chronic12.notna(),
            "only_raw_count_complete": count_raw.notna() & part.chronic12.isna(),
            "neither_count_complete": count_raw.isna() & part.chronic12.isna(),
        }
        require(sum(int(cond.sum()) for cond in conditions.values()) == len(part),
                "Chronic source categories must cover each sample exactly once")
        require(not (count_different & ~changed).any(), "A count difference requires a composition difference")
        partitions += [{"sample": name, "sample_n": len(part), "classification": key,
                        "n": int(cond.sum())} for key, cond in conditions.items()]
        summary = {
            "n": len(part), "raw_complete": int(count_raw.notna().sum()), "raw_incomplete": int(count_raw.isna().sum()),
            "harmonized_complete": int(part.chronic12.notna().sum()),
            "harmonized_incomplete": int(part.chronic12.isna().sum()),
            "any_item_state_differs_among_comparable_items": int(changed.sum()),
            "any_item_comparable_n": int(item_both_valid.ge(1).sum()),
            "any_baseline_dispute_flag_5_or_6": int(has_revised.sum()),
            "both_counts_available": int(comparable.sum()),
            "total_count_differs": int(count_different.sum()),
            "composition_differs_total_same": int((comparable & changed & ~count_different).sum()),
        }
        totals.append({"sample": name, **summary})
        summaries[name] = summary
    table = pd.concat(long_tables, ignore_index=True)
    write_csv(out / "A02_DISEASE_RAW_HARMONIZED_FLAG_CROSSTAB.csv", table)
    for group, keys in (("RAW_STATUS", ["raw_2011_code"]), ("HARMONIZED_STATUS", ["harmonized_code"]),
                        ("BASELINE_FLAGS", ["baseline_flag_code"])):
        dimensions = ["sample", "disease", "sample_n", *keys]
        write_csv(out / f"A02_{group}_DISTRIBUTION.csv", table.groupby(dimensions, sort=False, as_index=False).n.sum())
    write_csv(out / "A02_DISPUTE_TIMING_AND_ITEM_DIFFERENCES.csv", timing)
    write_csv(out / "A02_AGGREGATE_COUNT_CLASSES.csv", partitions)
    write_csv(out / "A02_COUNT_COMPARISON.csv", totals)
    extended = summaries["extended_595"]
    extended_frame = main.loc[main.B_M2_R]
    common = extended_frame.loc[extended_frame.chronic12_raw2011.notna()]
    groups = [{"group": group, "N": len(part), "events": int(part.Y.sum()),
               "communities": int(part.cluster.nunique())}
              for group in ("ALL", *GROUPS)
              for part in (common if group == "ALL" else common.loc[common.Bgroup.eq(group)],)]
    write_csv(out / "A02_E_COMMON_GROUPS.csv", groups)
    return summaries | {
        "trigger_eligible": bool(extended["raw_incomplete"] or extended["total_count_differs"]),
        "common_groups": groups,
        "wave_timing_limit": "Compatible 2013 flags corroborate a dispute in 2013; final baseline flags alone "
                             "cannot recover the timing of all revisions or identify the last updating wave.",
        "sources_are_distinct": "Raw count uses 12 doctor-diagnosis items in 2011 only. Harmonized Version D "
                                "uses retrospectively reconciled baseline status.",
    }


def check_do(raw):
    with ZipFile(raw / DO_ZIP) as z:
        names = z.namelist()
        require(names == ["H_CHARLS_D_do_file.do"], "Unexpected Harmonized construction bundle")
        lines = z.read(names[0]).decode("utf-8", errors="replace").splitlines()
    walk = [(i, line.strip()) for i, line in enumerate(lines, 1)
            if i < 5000 and "walk100a" in line.lower() and "db003" in line.lower()]
    require(len(walk) >= 6 and any("db001== 1 | db002==1" in line for _, line in walk),
            "Official construction rules do not corroborate normal skip and direct walking responses")
    checked = []
    for disease, h, f, r in DISEASES:
        harm = h[2:]
        flag = f[2:]
        direct = [(i, line.strip()) for i, line in enumerate(lines, 1)
                  if 4700 <= i <= 5050 and (f"r`wv'{harm} = 1 if {r}==1" in line or
                                             f"r`wv'{harm} = 0 if {r}==2" in line)]
        f_init = [(i, line.strip()) for i, line in enumerate(lines, 1)
                  if 4700 <= i <= 5050 and f"r`wv'{flag} = 0 if inlist({r},1,2)" in line]
        r2_retro = [(i, line.strip()) for i, line in enumerate(lines, 1)
                    if 6500 <= i <= 7050 and f"r`pre_wv'{flag}" in line and "replace" in line]
        require(len(direct) == 2 and len(f_init) == 1, f"Wave 1 construction do-file mapping requires review: {disease}")
        checked.append({"disease": disease, "wave1_status": h, "wave1_flag": f, "2011_item": r,
                        "codebook": "Version D printed pages 128-129 / PDF pages 129-130",
                        "baseline_construction_do_file_line_numbers": "; ".join(str(i) for i, _ in direct + f_init),
                        "wave2_retro_flag_do_file_line_numbers": "; ".join(str(i) for i, _ in r2_retro),
                        "raw_value_rule": "1 -> 1; 2 -> 0; any other code missing; all 12 items valid for count",
                        "harmonized_value_rule": "0=no; 1=yes; later disputes may update Wave 1 released status"})
    return walk, checked


def load_inputs(raw, reference, baseline):
    """Rebuild checked cohort and attach raw count. Pure in-memory return for authorized fits."""
    raw, reference, baseline = map(Path, (raw, reference, baseline))
    with import_baseline(baseline) as impl, TemporaryDirectory(prefix="audit11_input_", dir=baseline.parent) as temp:
        _, entry, _, _, NoFit, core = impl
        temporary = Path(temp)
        entry_result = entry.check(raw, reference, temporary / "entry")
        require(entry_result["passed"], f"1.0 historical entry checks failed: {entry_result['hard_blockers']}")
        with NoFit() as guard:
            frame, _ = reconstruct_locked(raw, reference, baseline, temporary / "cohort", impl)
            main, *_ = read_audit_fields(raw, frame, core)
            frame["chronic12_raw2011"] = main.chronic12_raw2011.reindex(frame.index)
        require(guard.calls == 0, "A model API was called within source-data loading")
        return frame


def write_plan(handoff, source_hashes):
    now = iso_time()
    text = f"""# ANALYSIS 1.1 审查后补充方案

登记时间 {now}。本方案在来源数据审计前登记。已查看 ANALYSIS-1.0 结果。当前授权以用户请求和本轮附件为依据。

## 数据和来源

唯一分析基线为 CHARLS_ANALYSIS_1_0.zip，SHA-256 为 `{ZIP_SHA}`。复用经哈希核验的 1.0 只读队列函数和完全相同的软件环境，用 NoFit 阻断估计。对照 MEMBER_LOCK、三份成员和字段 CSV、五套设计矩阵和结局／社区汇总。独立重读 2011 健康与功能模块及 Harmonized D 数据。保留 Stata 系统和扩展缺失。原始数据和既有估计保留原样。

## A01 步行

读取 Harmonized r1walk100a 与原始 DB001、DB002、DB003。以官方 ID_w1→ID 桥一对一连接。核对正式问卷、Version D 印刷页 101 及发布构造脚本的第一波规则。按原码识别拒答和不知道，.a 单列。记录 DB003 有效、上游正常跳题与发布值的任何不一致。只有 Harmonized .m、原 DB003 系统缺失、上游至少一个明确无法完成（码 4）且无上游无困难（码 1）时，归为“与第一波已知跳题问题相符”。其他无明确原因者保留为不可进一步归因。无个体访问轨迹时不证明具体路由。报告原值、交叉模式、四组缺失及扩展样本缺失的边际与互斥分布。

## A02 慢病

按 DA007 的 1–10、13、14 项分别核对同病 Harmonized 状态、基线争议旗标及 2013 年旗标。原始 1 为是、2 为否，其他值缺失。12 项全有效才求和。每种病的状态和旗标分开汇总，对照可比较状态、完整总数和互斥总体类别。基线旗标 5／6 表示后续争议；与同病 2013 旗标方向匹配时记录该次争议得到佐证。其余具体回改波次保持不可判定。派生原始慢病总数只用 2011 当时回答。

## 补充分析触发

原扩展 595 人内原始慢病总数不全或与 Harmonized 总数不同，触发审查后来源敏感性。共同样本为原扩展样本与原始 12 项完整者的交集，沿用其余条件。R1 用原扩展公式和 Harmonized 总数；若仍为相同 595 人且设计无变更，直接复用 1.0 S2 扩展。R2 仅替换慢病来源。两者矩阵相同时只保留一个估计；均完整且逐人总数一致时复用原结果。真正新增估计沿用已冻结社区聚类稳健 modified Poisson 和有限样本校正、正态 Wald 区间、双容差及独立矩阵方差核算。检查身份、必要组覆盖、矩阵秩、收敛、有限估计、方差和数值稳定性。稀疏、宽区间或跨 1 作为解释边界。发现实质错误时单独锁定证据、范围和最小修复；此处不预判来源差异为错误。所有新增模型在证据触发后再登记精确样本、公式与估计时间。

## 登记校验

| 来源 | SHA-256 |
| --- | --- |
"""
    text += "".join(f"| {key} | `{value}` |\n" for key, value in source_hashes.items())
    path = handoff / "ADDENDUM_PLAN_v1.1.md"
    if path.exists():
        prior = path.read_text(encoding="utf-8")
        require(all(value in prior for value in source_hashes.values()),
                "Existing audit plan has a different code or input hash; create explicit revision record before rerun")
        return
    path.write_text(text, encoding="utf-8")


def run(raw, stage, archive, original_run):
    baseline, reference = stage / "baseline", stage / "historical_reference"
    handoff = stage / "handoff"
    out = handoff / "SOURCE_AUDIT"
    out.mkdir(parents=True, exist_ok=True)
    identity = verify_baseline(archive, baseline, original_run)
    source_hashes = {"audit11.py": sha(Path(__file__)),
                     **{p.as_posix(): sha(raw / p) for p in (RAW_HEALTH, HARMONIZED, DO_ZIP, DOC, CODEBOOK)}}
    write_plan(handoff, source_hashes)
    started = iso_time()
    with import_baseline(baseline) as impl, TemporaryDirectory(prefix="source_audit_", dir=stage) as temp:
        _, entry, _, _, NoFit, core = impl
        temp = Path(temp)
        reentry = entry.check(raw, reference, temp / "entry")
        require(reentry["passed"], f"Historical entry blocked: {reentry['hard_blockers']}")
        prior = read_json(baseline / "ENTRY_CHECK.json")
        require(reentry["raw_inputs"] == prior["raw_inputs"]
                and all(reentry["archives"][key]["archive_sha256"] == old["archive_sha256"]
                        for key, old in prior["archives"].items()),
                "Raw input or historical package hashes differ from 1.0")
        walk_do, mapping = check_do(raw)
        with NoFit() as guard:
            frame, members = reconstruct_locked(raw, reference, baseline, temp / "cohort", impl)
            main, raw_codes, h_codes, metadata, linkage = read_audit_fields(raw, frame, core)
            a01 = write_walk(main, raw_codes, h_codes, out)
            a02 = chronic_source(main, raw_codes, h_codes, out)
            require(guard.calls == 0, "A model API was called within source audit")
        write_csv(out / "FIELD_METADATA.csv", metadata)
        write_csv(out / "A02_DISEASE_MAPPING.csv", mapping)
        write_csv(out / "A01_OFFICIAL_CONSTRUCTION_LOCATIONS.csv",
                  [{"source": DO_ZIP.as_posix(), "line": i, "code": code} for i, code in walk_do])
        write_json(out / "SOURCE_LINKAGE_QA.json", linkage)
        synthetic = tests()
        write_json(out / "SYNTHETIC_BOUNDARY_TESTS.json", synthetic)
        require(all(sha(raw / p) == source_hashes[p.as_posix()] for p in (RAW_HEALTH, HARMONIZED, DO_ZIP, DOC, CODEBOOK)),
                "Audit input changed during execution")
        supplemental_inputs = [{"relative_source": p.as_posix(), "sha256": source_hashes[p.as_posix()],
                                "role": role, "listed_in_1_0": p != DO_ZIP}
                               for p, role in ((RAW_HEALTH, "DB001–DB003 and 12 doctor-diagnosis items"),
                                               (HARMONIZED, "walk100a, baseline states/flags and 2013 corroboration flags"),
                                               (DO_ZIP, "published construction rules, text inspection only"),
                                               (DOC, "2011 questionnaire DA007 and functional limitations routes"),
                                               (CODEBOOK, "Version D codebook printed pp. 101, 128–129"))]
        write_csv(out / "ADDENDUM_INPUTS_SHA256.csv", supplemental_inputs)
        write_json(handoff / "ENTRY_AND_MEMBER_QA_v1.1.json", {
            "status": "PASS", "baseline": identity, **members,
            "source_audit_started": started, "source_audit_finished": iso_time(),
            "frozen_18_inputs_verified": len(reentry["raw_inputs"]),
            "verified_frozen_input_digests": reentry["raw_inputs"],
            "historical_archives": {name: {k: v for k, v in values.items() if k != "file_hashes"}
                                    for name, values in reentry["archives"].items()},
            "source_audit_fit_calls": guard.calls, "row_level_output_written": False,
        })
    summary = {"A01": a01, "A02": a02, "fit_calls": 0, "started": started, "finished": iso_time()}
    write_json(out / "SOURCE_SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


def tests():
    candidates = pd.DataFrame([["1"] + ["2"] * 11, [".d"] + ["2"] * 11,
                               ["2", "1"] + ["2"] * 10, ["3"] + ["2"] * 11])
    counts = raw_count(candidates)
    results = {
        "valid_codes_and_complete_item_count": bool(counts.iloc[0] == counts.iloc[2] == 1),
        "dont_know_and_unexpected_code_remain_missing": bool(counts.iloc[[1, 3]].isna().all()),
        "composition_change_can_preserve_count": bool(not candidates.iloc[0].equals(candidates.iloc[2])),
        "route_pattern_conservative": walk_reason(".m", "4", ".", ".")[0] == "consistent_with_known_wave1_routing_problem"
        and walk_reason(".m", "2", "3", ".")[0] == "cannot_further_attribute"
        and walk_reason(".m", "4", "1", ".")[0] == "cannot_further_attribute",
        "missing_code_preserved": stata_code(StataMissingValue(103)) == ".b",
    }
    require(all(results.values()), f"Source boundary tests failed: {results}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--baseline-zip", type=Path, required=True)
    parser.add_argument("--original-run", type=Path)
    args = parser.parse_args()
    run(args.raw.resolve(), args.stage.resolve(), args.baseline_zip.resolve(), args.original_run)
