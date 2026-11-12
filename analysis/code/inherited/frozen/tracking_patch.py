
"""2015 cause-only attribution, with one-to-one raw/Harmonized source linkage."""
import json
from zipfile import ZipFile

import pandas as pd
from pandas.io.stata import StataReader

from audit_core import code_number, exact_index, string_codes

HARMONIZED_CODES = {
    0: "0.inap.", 1: "1.resp, alive", 4: "4.nr, alive",
    5: "5.nr, died this wv", 6: "6.nr, died prev wv",
    7: "7.nr, dropped from samp", 9: "9.nr, dk if alive or died",
}
RAW_DEATH_CODES = {"0": "0 Alive", "1": "1 Died"}
T1_DEATH_CURRENT = "DEATH_BY_T1_CURRENT_WAVE"
T1_DEATH_PREVIOUS = "DEATH_BY_T1_PREVIOUS_WAVE"
T1_DEATH_RAW_ONLY = "DEATH_BY_T1_RAW_ONLY"


def raw_index(frame, label, qc):
    indexed = exact_index(frame, "ID", label, qc)
    if not indexed.index.str.fullmatch(r"[0-9]{12}").all():
        raise ValueError(f"{label}: noncanonical key shape")
    return indexed


def align_sources(d, sample, harmonized, demographics, health):
    """Reindex each original source to all baseline keys. Never select by T1."""
    named = (
        ("baseline", d), ("Sample_Infor", sample), ("Harmonized D", harmonized),
        ("2015 demographics", demographics), ("2015 health", health),
    )
    for name, frame in named:
        if not frame.index.is_unique:
            raise ValueError(f"{name}: duplicate key; no join performed")
    demo_matched = d.index.isin(demographics.index)
    if not (d.in_demo1.eq(True).fillna(False).to_numpy() == demo_matched).all():
        raise ValueError("2015 demographics keys disagree with v1.2 reconstructed flags")
    health_matched = d.index.isin(health.index)
    if not (d.in_health1.eq(True).fillna(False).to_numpy() == (health_matched & demo_matched)).all():
        raise ValueError("2015 health keys disagree with v1.2 demographics-restricted flags")
    sample_matched = d.index.isin(sample.index)
    harmonized_matched = d.index.isin(harmonized.index)
    raw_codes = string_codes(sample.died).reindex(d.index)
    r3_codes = string_codes(harmonized.r3iwstat).reindex(d.index)
    died_full = code_number(raw_codes)
    r3_full = code_number(r3_codes)
    old_death_equal = d.dead1.eq(died_full) | (d.dead1.isna() & died_full.isna())
    if not old_death_equal.all():
        raise ValueError("Sample_Infor raw death flags disagree with v1.2 reconstruction")
    # The archived 2015 wave frame included the union of demographics and sample.
    old_tracking_full = r3_full.where(demo_matched | sample_matched)
    old_tracking_equal = d.iwstat1.eq(old_tracking_full) | (d.iwstat1.isna() & old_tracking_full.isna())
    if not old_tracking_equal.all():
        raise ValueError("v1.2 iwstat1 coverage is not explained by its wave-frame index")
    return d.assign(
        t1_sample_matched=sample_matched, t1_harmonized_matched=harmonized_matched,
        t1_demo_matched_raw=demo_matched, t1_health_matched_raw=health_matched,
        t1_raw_died=raw_codes.fillna("NO_MATCHING_ROW"),
        t1_r3iwstat=r3_codes.fillna("NO_MATCHING_ROW"),
        t1_died_full=died_full, r3iwstat_full=r3_full,
    )


def classify_sources(d):
    """Use positive evidence and preserve mortality conflicts explicitly."""
    raw, h = d.t1_died_full, d.r3iwstat_full
    demo, health = d.t1_demo_matched_raw, d.t1_health_matched_raw
    interview = demo | health
    h_dead, raw_dead = h.isin([5, 6]), raw.eq(1)
    h_alive, raw_alive = h.isin([1, 4]), raw.eq(0)
    conflict = (h_dead | raw_dead) & (interview | h_alive | raw_alive)
    confirmed = (h_dead | raw_dead) & ~conflict
    reason = pd.Series("NO_T1_TRACKING_SUBTYPE", index=d.index, dtype="str")
    evidence = pd.Series("No decisive 2015 source status", index=d.index, dtype="str")

    reason.loc[h.eq(0)] = "NOT_IN_WAVE"
    evidence.loc[h.eq(0)] = "Harmonized r3iwstat=0, sample not yet entered"
    reason.loc[h.eq(7)] = "DROPPED_FROM_SAMPLE"
    evidence.loc[h.eq(7)] = "Harmonized r3iwstat=7, dropped from sample in DTA label"
    reason.loc[h.eq(9)] = "NONRESPONSE_VITAL_UNKNOWN"
    evidence.loc[h.eq(9)] = "Harmonized r3iwstat=9, nonresponse and vital status unknown"
    reason.loc[raw_alive] = "NONRESPONSE_KNOWN_ALIVE"
    evidence.loc[raw_alive] = "Sample_Infor died=0, labeled alive; no matched original interview"
    reason.loc[h.eq(4)] = "NONRESPONSE_KNOWN_ALIVE"
    evidence.loc[h.eq(4)] = "Harmonized r3iwstat=4, nonresponse and known alive"

    reason.loc[~demo & health] = "RAW_HEALTH_WITHOUT_DEMOGRAPHICS"
    evidence.loc[~demo & health] = "Original 2015 health matched without a demographics match"
    reason.loc[~demo & h.eq(1)] = "HARMONIZED_RESPONDED_MODULE_MISMATCH"
    evidence.loc[~demo & h.eq(1)] = "Harmonized r3iwstat=1, no matching original demographics interview"
    reason.loc[demo] = "INTERVIEW_OBSERVED"
    evidence.loc[demo] = "Original 2015 demographics interview matched"
    reason.loc[demo & ~health & h.eq(1)] = "HARMONIZED_RESPONDED_HEALTH_MISMATCH"
    evidence.loc[demo & ~health & h.eq(1)] = "Harmonized r3iwstat=1 and original demographics matched; health unmatched"

    reason.loc[confirmed & ~h_dead] = T1_DEATH_RAW_ONLY
    evidence.loc[confirmed & ~h_dead] = "Sample_Infor died=1; Harmonized death timing subtype unavailable"
    reason.loc[confirmed & h.eq(5)] = T1_DEATH_CURRENT
    evidence.loc[confirmed & h.eq(5)] = "Harmonized r3iwstat=5, death recorded in current wave"
    reason.loc[confirmed & h.eq(6)] = T1_DEATH_PREVIOUS
    evidence.loc[confirmed & h.eq(6)] = "Harmonized r3iwstat=6, died before previous wave; no individual date"
    both = confirmed & h_dead & raw_dead
    evidence.loc[both] = evidence.loc[both] + "; Sample_Infor died=1 agrees"

    reason.loc[conflict] = "SOURCE_CONFLICT"
    evidence.loc[conflict] = "2015 death status conflicts with recorded alive status or original interview"
    primary = reason.where(~demo | conflict)
    return d.assign(t1_tracking_reason=primary, t1_tracking_status=reason,
                    t1_tracking_evidence=evidence, t1_source_conflict=conflict,
                    t1_confirmed_death=confirmed)


def link_tracking(d, inp):
    """Return in-memory augmented records and aggregate provenance."""
    harmonized_zip = inp.register(
        inp.root / "Harmonized CHARLS" / "H_CHARLS_D_Data.zip",
        "Harmonized D linkage and covariates",
    )
    codebook = inp.register(
        inp.root / "Harmonized CHARLS" / "Harmonized_CHARLS_D.pdf",
        "Verified v1.2 Harmonized D codebook, printed p30 / physical p31",
    )
    with ZipFile(harmonized_zip) as z, z.open("H_CHARLS_D_Data.dta") as source:
        with StataReader(source, convert_categoricals=False, convert_missing=True) as r:
            variables = r.variable_labels()
            label_keys = dict(zip(variables, r._lbllist))
            labels = r.value_labels()[label_keys["r3iwstat"]]
            h_frame = r.read(columns=["ID", "r3iwstat"])
    if {code: labels.get(code) for code in HARMONIZED_CODES} != HARMONIZED_CODES:
        raise ValueError("Harmonized D interview status labels differ from the verified release")
    h = raw_index(h_frame, "PATCH full Harmonized D r3iwstat", inp.qc)
    s_frame = inp.read(2015, "sample_infor", ["ID", "died"])
    s = raw_index(s_frame, "PATCH raw 2015 Sample_Infor.died", inp.qc)
    raw_meta = inp.metadata[(2015, "sample_infor", "died")]
    if json.loads(raw_meta["dta_code_labels"]) != RAW_DEATH_CODES:
        raise ValueError("Sample_Infor death labels differ from the verified release")
    demo = raw_index(inp.read(2015, "demographic_background", ["ID"]),
                     "PATCH raw 2015 demographics keys", inp.qc)
    health = raw_index(inp.read(2015, "health_status_and_functioning", ["ID"]),
                       "PATCH raw 2015 health keys", inp.qc)
    patch = classify_sources(align_sources(d, s, h, demo, health))

    def totals(q):
        return {
            "n": len(q), "harmonized_matches": int(q.t1_harmonized_matched.sum()),
            "sample_matches": int(q.t1_sample_matched.sum()),
            "demographics_matches": int(q.t1_demo_matched_raw.sum()),
            "health_matches": int(q.t1_health_matched_raw.sum()),
            "mortality_source_conflicts": int(q.t1_source_conflict.sum()),
            "confirmed_deaths": int(q.t1_confirmed_death.sum()),
            "tracking_status_counts": {
                str(k): int(v) for k, v in sorted(q.t1_tracking_status.value_counts().items())
            },
        }

    source_meta = {
        "source_scope": "2015 sources only; no inference from 2018 death",
        "linkage": "exact one-to-one string ID reindex to complete v1.2 baseline",
        "harmonized_path": str(harmonized_zip), "harmonized_variable": "r3iwstat",
        "harmonized_labels": {str(k): v for k, v in labels.items()},
        "codebook_path": str(codebook), "codebook_physical_page": 31,
        "codebook_printed_page": 30,
        "official_url": "https://charls.pku.edu.cn/en/team/Codebook.pdf",
        "source_2015_sample_metadata": raw_meta,
        "source_row_counts": {"harmonized": len(h), "sample_infor": len(s),
                              "demographics": len(demo), "health": len(health)},
        "full_baseline": totals(patch),
        "B_start": totals(patch.loc[patch.B_start]),
        "B_start_without_t1_demographics": totals(patch.loc[patch.B_start & ~patch.t1_demo_matched_raw]),
    }
    return patch, source_meta


def tracking_synthetic_tests():
    """Cause tests with manufactured rows. No model estimation."""
    d = pd.DataFrame({
        "t1_died_full": [None, 1, None, None, 0, 1, 1, None, None, None, 1, None],
        "r3iwstat_full": [6, 5, 4, 9, 6, 1, 5, 1, 1, 7, None, None],
        "t1_demo_matched_raw": [False] * 6 + [True, False, True, False, False, False],
        "t1_health_matched_raw": [False] * 6 + [True, False, False, False, False, False],
    }, index=pd.Index([f"SYNTHETIC_{n}" for n in range(12)], name="ID"))
    out = classify_sources(d)
    tests = [
        ("T1_PREVIOUS_DEATH_WITHOUT_RAW_ROW", out.t1_tracking_status.iloc[0] == T1_DEATH_PREVIOUS,
         "Harmonized code 6 confirms prior death with no 2015 raw row."),
        ("T1_CURRENT_WAVE_DEATH", out.t1_tracking_status.iloc[1] == T1_DEATH_CURRENT,
         "Harmonized code 5 plus raw death flag 1 counts as one death."),
        ("T1_ALIVE_NONRESPONSE", out.t1_tracking_status.iloc[2] == "NONRESPONSE_KNOWN_ALIVE",
         "Code 4 means known alive and nonresponse."),
        ("T1_UNKNOWN_VITAL_NONRESPONSE", out.t1_tracking_status.iloc[3] == "NONRESPONSE_VITAL_UNKNOWN",
         "Code 9 remains unknown."),
        ("T1_SOURCE_CONFLICT", out.t1_source_conflict.iloc[[4,5,6]].all()
         and not out.t1_confirmed_death.iloc[[4,5,6]].any(),
         "Opposed source status or interview yields conflict, not confirmed death."),
        ("T1_RESPONDED_MODULE_MISMATCH", out.t1_tracking_status.iloc[7] == "HARMONIZED_RESPONDED_MODULE_MISMATCH"
         and out.t1_tracking_status.iloc[8] == "HARMONIZED_RESPONDED_HEALTH_MISMATCH",
         "Harmonized response with a missing original module remains distinguishable."),
        ("T1_OTHER_OR_MISSING", out.t1_tracking_status.iloc[9:].tolist()
         == ["DROPPED_FROM_SAMPLE", T1_DEATH_RAW_ONLY, "NO_T1_TRACKING_SUBTYPE"],
         "Other release code, raw-only death, and no source remain separate."),
        ("T1_CAUSES_EXCLUSIVE", out.t1_tracking_status.notna().all()
         and sum(out.t1_tracking_status.value_counts()) == len(out),
         "Exactly one source disposition per synthetic key."),
    ]
    duplicate = d.iloc[[0, 0]]
    try:
        align_sources(d, duplicate, d, d, d)
    except ValueError as exc:
        duplicate_rejected = "duplicate key" in str(exc)
    else:
        duplicate_rejected = False
    tests.append(("T1_DUPLICATE_KEY_REJECTED", duplicate_rejected,
                  "Duplicate source keys stop linkage rather than multiplying records."))
    return [
        {"phase": "SYNTHETIC", "test": label, "status": "PASS" if ok else "FAIL", "evidence": message}
        for label, ok, message in tests
    ]

