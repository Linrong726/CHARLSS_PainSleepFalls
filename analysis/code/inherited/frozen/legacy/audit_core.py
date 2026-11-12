
"""Read-only CHARLS preflight. Never writes respondent-level data."""
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from pandas.io.stata import StataReader

LOG = logging.getLogger("preflight")
WAVE_NUMBER = {2011: 1, 2013: 2, 2015: 3}
ITEMS = {
    2011: dict(pain="da041", sleep="dc015", fall="da023", new=None,
               proxy=["proxy", "db032"], binary_pain=True),
    2013: dict(pain="wb16", sleep="dc015", fall="da023", new=None,
               proxy=["db032"], binary_pain=False),
    2015: dict(pain="da041", sleep="dc015", fall="da023", new=None,
               proxy=["proxy", "db032"], binary_pain=True),
    2018: dict(pain="da041_w4", sleep="dc015", fall="da023_w4", new="da023",
               proxy=["db032"], binary_pain=False),
    2020: dict(pain="da027", sleep="dc022", fall="da022", new="da023",
               proxy=["proxy_5", "proxy", "db045"], binary_pain=False),
}
CHRONIC = "hibpe dyslipe diabe cancre lunge livere hearte stroke kidneye digeste arthre asthmae".split()
MOOD = [f"dc{i:03d}" for i in range(9,19) if i != 15]
M1 = ["age", "sex", "education", "rural", "prior_fall"]
M2 = M1 + ["mood9", "chronic12", "walk100", "vision", "memory"]
GROUPS = {"11":"both_persist","10":"sleep_only","01":"pain_only","00":"both_resolve"}
A_GROUPS = ["11","10","01","00"]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(path):
    with open(path, "rb") as fp:
        return hashlib.file_digest(fp, "sha256").hexdigest()


def dta(root, wave, stem):
    folder, = root.glob(f"{wave} CHARLS Wave*")
    path, = [p for p in folder.rglob("*.dta") if p.stem.lower() == stem.lower()]
    return path


def string_codes(col):
    return col.map(str).replace({"nan": ".", "<NA>": ".", "None": "."})


def code_number(col):
    return pd.to_numeric(string_codes(col), errors="coerce")


def numeric_subset(col, allowed):
    n = code_number(col)
    return n.where(n.isin(allowed)).astype(float)


def binary(col):
    return numeric_subset(col, [1,2]).map({1:1.,2:0.})


def pain(col, is_binary):
    if is_binary:
        return binary(col)
    val = numeric_subset(col, range(1,6))
    return val.gt(1).astype(float).where(val.notna())


def sleep(col):
    val = numeric_subset(col, [1,2,3,4])
    return val.ge(3).astype(float).where(val.notna())


def mood9(raw):
    valid = raw[MOOD].apply(lambda s: numeric_subset(s, [1,2,3,4]))
    score = valid - 1
    score[["dc013","dc016"]] = 3 - score[["dc013","dc016"]]
    return score.sum(axis=1, min_count=9)


def exact_index(frame, key, tag, qc):
    vals = frame[key]
    assert vals.map(lambda s: isinstance(s, str)).all(), tag
    present = vals.ne("")
    dup = vals[present].duplicated(keep=False)
    qc.append(dict(stage=tag, rows=len(frame), unique=int(vals[present].nunique()),
                   missing_key=int((~present).sum()), duplicate_rows=int(dup.sum()),
                   id_lengths="|".join(str(i) for i in sorted(vals[present].str.len().unique())),
                   status="PASS" if present.all() and not dup.any() else "FAIL"))
    assert present.all() and not dup.any(), tag
    return frame.set_index(key, verify_integrity=True)


def join(left, right, tag, qc):
    counts = dict(stage=tag, rows=len(left), right_rows=len(right),
                  overlap=int(left.index.isin(right.index).sum()),
                  left_only=int((~left.index.isin(right.index)).sum()),
                  right_only=int((~right.index.isin(left.index)).sum()))
    result = left.join(right, how="left", validate="one_to_one")
    assert len(result) == len(left), tag
    qc.append(counts | dict(out_rows=len(result), status="PASS"))
    return result


def month_bounds(year, month):
    t = pd.to_datetime({"year":pd.to_numeric(year, errors="coerce"),
                        "month":pd.to_numeric(month, errors="coerce"),
                        "day":1}, errors="coerce")
    return t, t + pd.offsets.MonthEnd(0)


def last_bounds(value):
    parts = value.str.extract(r"^([0-9]{4})年([0-9]{2})月$")
    return month_bounds(parts[0], parts[1])


def strict_order(earlier_hi, later_lo):
    return earlier_hi.notna() & later_lo.notna() & earlier_hi.lt(later_lo)


def outcome_status(d, has_verified_anchor):
    # Run all tests even where exposure measurement blocks inference.
    status = pd.Series("NO_VALID_Y_UNCLASSIFIED", index=d.index, dtype="str")
    health = d["in_health2"].eq(True)
    demo = d["in_demo2"].eq(True)
    dead = d["dead2"].eq(1)
    status.loc[~health & demo] = "HEALTH_MODULE_UNAVAILABLE"
    status.loc[~health & ~demo] = "NO_INTERVIEW_OR_UNKNOWN_DISPOSITION"
    # A confirmed record of death takes precedence over absence or other values.
    status.loc[dead] = "DEATH"
    current = health & ~dead
    date_order = (d["in_demo1"].eq(True) & d["in_demo2"].eq(True)
                  & strict_order(d["date_hi0"],d["date_lo1"])
                  & strict_order(d["date_hi1"],d["date_lo2"]))
    if has_verified_anchor:
        returned = current & d["route2"].eq(1)
        new = current & d["route2"].eq(2)
        exact_month = d["last_lo2"].notna() & d["last_lo2"].eq(d["date_lo1"])
        older = d["last_hi2"].notna() & d["last_hi2"].lt(d["date_lo1"])
        status.loc[current] = "TIME_UNCERTAIN_ROUTE"
        status.loc[new] = "NEW_LIFETIME_STRUCTURAL"
        status.loc[returned] = "TIME_UNCERTAIN_ANCHOR"
        status.loc[returned & older] = "TIME_UNCERTAIN_OVERLAP"
        status.loc[returned & exact_month & ~date_order] = "TIME_UNCERTAIN_DATE_ORDER"
        anchored = returned & exact_month & date_order
        # Both fall routes must be mutually exclusive in the raw release.
        anchored &= ~d["new_fall2"].notna()
        status.loc[anchored] = "FALL_ITEM_BLANK_OR_UNDIFFERENTIATED"
        status.loc[anchored & d["fall_code2"].isin([".d","8","997"])] = "FALL_DONT_KNOW"
        status.loc[anchored & d["fall_code2"].isin([".r","9","999"])] = "FALL_REFUSAL"
        status.loc[anchored & d["fall2"].notna() & d["proxy_unknown2"].eq(True)] = "PROXY_STATUS_UNCERTAIN"
        status.loc[anchored & d["proxy2"].eq(True)] = "PROXY_OUTCOME"
        valid = anchored & ~d["proxy2"].eq(True) & ~d["proxy_unknown2"].eq(True)
        status.loc[valid & d["fall2"].notna()] = "VALID_Y"
    else:
        status.loc[current] = "TIME_UNCERTAIN_ANCHOR_NOT_RELEASED"
        status.loc[current & d["route2"].eq(2)] = "NEW_LIFETIME_STRUCTURAL"
    return status, d["fall2"].where(status.eq("VALID_Y"))


class Inputs:
    def __init__(self, root):
        self.root = root
        self.manifest = {}
        self.metadata = {}
        self.qc = []

    def register(self, path, role):
        if path not in self.manifest:
            st = path.stat()
            self.manifest[path] = dict(path=str(path), role=role, size_bytes=st.st_size,
                                       mtime_utc=datetime.fromtimestamp(st.st_mtime,timezone.utc).isoformat(),
                                       sha256_before=sha(path))
        return path

    def read(self, wave, stem, cols):
        p = self.register(dta(self.root,wave,stem), f"{wave} {stem}")
        with StataReader(p, convert_categoricals=False, convert_missing=True) as r:
            labels = r.variable_labels()
            values = r.value_labels(); label_keys = dict(zip(labels,r._lbllist))
            selected = [k for k in dict.fromkeys(cols) if k in labels]
            self.manifest[p].update(stata_format=r._format_version, source_timestamp=r._time_stamp)
            version_col = next((k for k in labels if k.lower()=="versionid"), None)
            df = r.read(columns=selected+([version_col] if version_col else []))
            if version_col:
                self.manifest[p]["release_version"] = "|".join(sorted(string_codes(df.pop(version_col)).unique()))
            for k in cols:
                if k not in labels:
                    self.metadata[(wave,stem,k)] = dict(path=str(p),year=wave,module=stem,
                                                         variable=k,status="ABSENT_IN_RELEASE")
                    continue
                if k.lower().endswith("id") or k == "ziwtime":
                    continue
                counts = string_codes(df[k]).value_counts()
                self.metadata[(wave,stem,k)] = dict(
                    path=str(p), year=wave, module=stem, variable=k, label=labels[k],
                    dta_code_labels=json.dumps({str(i):j for i,j in values.get(label_keys[k],{}).items()},ensure_ascii=False),
                    dta_frequency=json.dumps({str(i):int(j) for i,j in counts.items()},ensure_ascii=False)
                                  if len(counts) <= 24 else None, status="OBSERVED")
        return df

    def harmonized(self):
        p = self.register(self.root/"Harmonized CHARLS"/"H_CHARLS_D_Data.zip", "Harmonized D linkage and covariates")
        cols = ["ID","ID_w1","ragender"]
        for w in WAVE_NUMBER.values():
            cols += [f"r{w}agey",f"h{w}rural",f"r{w}walk100a",f"r{w}iwstat",
                     f"r{w}iwy",f"r{w}iwm"] + [f"r{w}{c}" for c in CHRONIC]
        cols += ["r4iwstat","r4iwy","r4iwm"]
        with ZipFile(p) as z, z.open("H_CHARLS_D_Data.dta") as f:
            with StataReader(f, convert_categoricals=False, convert_missing=True) as r:
                labels = r.variable_labels()
                values = r.value_labels(); label_keys = dict(zip(labels,r._lbllist))
                self.manifest[p].update(stata_format=r._format_version, source_timestamp=r._time_stamp)
                for k in cols[2:]:
                    self.metadata[(0,"Harmonized_D",k)] = dict(path=str(p),year=0,module="Harmonized_D",
                        variable=k,label=labels[k],status="OBSERVED",
                        dta_code_labels=json.dumps({str(i):j for i,j in values.get(label_keys[k],{}).items()},ensure_ascii=False))
                df = r.read(columns=cols)
        return exact_index(df,"ID","Harmonized_D canonical IDs",self.qc)

    def verify(self):
        records = []
        for p, rec in self.manifest.items():
            after = sha(p)
            records.append(rec | dict(sha256_after=after, unchanged=rec["sha256_before"]==after))
        return records


def canonical(frame, wave, bridge, tag, qc):
    base = exact_index(frame,"ID",tag,qc)
    expected = 11 if wave == 2011 else 12
    assert base.index.str.fullmatch(rf"[0-9]{{{expected}}}").all(), tag
    if wave == 2011:
        mapped = base.index.map(bridge)
        missing = mapped.isna()
        qc.append(dict(stage=tag+" exact ID_w1→ID", rows=len(base), overlap=int((~missing).sum()),
                       left_only=int(missing.sum()), status="PASS"))
        base.index = pd.Index([f"W1:{old}" if absent else new
                              for old,new,absent in zip(base.index,mapped,missing)],name="ID")
        assert base.index.is_unique
    return base


def load_wave(inp, wave, harmonized, bridge):
    v = ITEMS[wave]
    mod = "health_status_and_functioning"
    vars_ = ["ID","communityID",v["pain"],v["fall"],"da024"] + v["proxy"]
    if wave >= 2013:
        vars_ += ["xrtype","ziwtime"]
    if v["new"]:
        vars_ += [v["new"]]
    if wave != 2018:
        vars_ += [v["sleep"]]
    if wave in WAVE_NUMBER:
        vars_ += MOOD + ["da032","da033","dc004"]
    if wave in (2013,2020):
        vars_ += ["da041" if wave==2013 else "dc015"]
    raw = canonical(inp.read(wave,mod,vars_),wave,bridge,f"{wave} health",inp.qc)
    if wave == 2018:
        cog = canonical(inp.read(wave,"Cognition",["ID",v["sleep"]]),wave,bridge,
                        "2018 Cognition",inp.qc)
        raw = join(raw,cog.assign(in_sleep_module=True),"2018 health×cognition",inp.qc)
    else:
        raw["in_sleep_module"] = True
    d = pd.DataFrame(index=raw.index)
    d["in_health"] = True
    d["in_sleep_module"] = raw["in_sleep_module"]
    d["cluster"] = raw["communityID"].where(raw["communityID"].ne(""))
    d["pain_code"] = string_codes(raw[v["pain"]])
    d["sleep_code"] = string_codes(raw[v["sleep"]])
    d["fall_code"] = string_codes(raw[v["fall"]])
    d["new_fall"] = binary(raw[v["new"]]) if v["new"] else np.nan
    d["treated_count"] = code_number(raw["da024"])
    d["fall"] = binary(raw[v["fall"]])
    db_proxy = code_number(raw["db032"]).eq(4) if "db032" in raw else False
    d["proxy"] = db_proxy
    d["proxy_unknown"] = False
    if wave in (2011,2015):
        proxy = code_number(raw["proxy"])
        d["proxy"] |= proxy.eq(1)
        d["proxy_unknown"] = ~proxy.isin([0,1,2]) & ~d["proxy"]
    if wave in (2013,2018):
        d["proxy_unknown"] = ~code_number(raw["db032"]).isin([1,2,3,4])
    if wave == 2020:
        d["proxy"] = code_number(raw["proxy_5"]).eq(1) | code_number(raw["db045"]).eq(4)
        d["proxy_unknown"] = code_number(raw["proxy"]).eq(1) & ~code_number(raw["proxy_5"]).isin([1,2])
    self_or_assisted = ~d["proxy"] & ~d["proxy_unknown"]
    d["p"] = pain(raw[v["pain"]], v["binary_pain"]).where(self_or_assisted)
    d["s"] = sleep(raw[v["sleep"]]).where(self_or_assisted)
    d["route"] = code_number(raw["xrtype"]) if "xrtype" in raw else np.nan
    if wave in (2013,2015):
        d["route"] = d["route"].map({2:1.,1:2.})
    if "ziwtime" in raw:
        p = dta(inp.root,wave,mod)
        inp.metadata[(wave,mod,"ziwtime")] = dict(path=str(p),year=wave,module=mod,
            variable="ziwtime",label="R's LAST IW Time",status="OBSERVED")
        d["last_lo"],d["last_hi"] = last_bounds(raw["ziwtime"])
    else:
        d["last_lo"],d["last_hi"] = pd.NaT,pd.NaT

    dm = canonical(inp.read(wave,"demographic_background",["ID","bd001","bd001_w2_4","bd001_w2_3","zbd001",
              "rgender","ba000_w2_3"]),wave,bridge,f"{wave} demographics",inp.qc)
    d = join(dm.assign(in_demo=True),d,f"{wave} demographics×health",inp.qc)
    # Keep demographics as the cohort denominator and document any health-only IDs.
    tracking_stem = {2011:"weight",2013:"weights"}.get(wave,"sample_infor")
    tr_raw = inp.read(wave,tracking_stem,["ID","iyear","imonth","died","INDV_L_Died","crosssection"])
    width = 11 if wave==2011 else 12
    shape = tr_raw.ID.str.fullmatch(rf"[0-9]{{{width}}}")
    inp.qc.append(dict(stage=f"{wave} tracking ID format prefilter",rows=len(tr_raw),
                       overlap=int(shape.sum()),left_only=int((~shape).sum()),status="OBSERVED"))
    tr = canonical(tr_raw.loc[shape],
                   wave,bridge,f"{wave} sample/dates",inp.qc)
    tr["date_lo"],tr["date_hi"] = month_bounds(tr["iyear"],tr["imonth"])
    death_field = {2013:"INDV_L_Died"}.get(wave,"died")
    tr["dead"] = numeric_subset(tr[death_field],[0,1]) if death_field in tr else np.nan
    d = d.join(tr[["date_lo","date_hi","dead"]],how="outer",validate="one_to_one")
    if wave in (2013,2020):
        ex_stem = {2013:"exit_interview",2020:"exit_module"}[wave]
        ex = canonical(inp.read(wave,ex_stem,["ID"]),wave,bridge,f"{wave} exit",inp.qc)
        d["exit_interview"] = d.index.isin(ex.index)
        d.loc[d["exit_interview"],"dead"] = 1.
    else:
        d["exit_interview"] = False

    if wave in WAVE_NUMBER:
        w = WAVE_NUMBER[wave]
        h = harmonized.reindex(d.index)
        d["age"] = numeric_subset(h[f"r{w}agey"],range(0,121))
        d["sex"] = numeric_subset(d["rgender"] if "rgender" in d else d["ba000_w2_3"],[1,2])
        ed = numeric_subset(d["bd001"],range(1,12)) if "bd001" in d else pd.Series(np.nan,index=d.index)
        d["education"] = education_levels(ed)
        d["rural"] = numeric_subset(h[f"h{w}rural"],[0,1])
        d["prior_fall"] = d["fall"].where(~d["proxy"].eq(True) & ~d["proxy_unknown"].eq(True))
        # Subjective memory and distance vision, each kept as five ordered levels.
        d["memory"] = numeric_subset(raw["dc004"],range(1,6)).where(self_or_assisted)
        d["vision"] = numeric_subset(raw["da033"],range(1,6)).mask(code_number(raw["da032"]).eq(2),5)
        d["walk100"] = numeric_subset(h[f"r{w}walk100a"],[0,1])
        d["mood9"] = mood9(raw).where(self_or_assisted)
        chronic = h[[f"r{w}{c}" for c in CHRONIC]].apply(lambda s: numeric_subset(s,[0,1]))
        d["chronic12"] = chronic.sum(axis=1,min_count=len(CHRONIC))
        d["iwstat"] = code_number(h[f"r{w}iwstat"])
    elif wave == 2018:
        d["iwstat"] = code_number(harmonized.reindex(d.index)["r4iwstat"])
    else:
        d["iwstat"] = np.nan
    return d


def education_levels(ed):
    return ed.map({1:0,2:0,3:0,4:0,5:0,6:1,7:1,8:2,9:2,10:2,11:2})


def followup_education(waves):
    """Rebuild released current/unchanged education with past data only."""
    w2,w3 = waves[2013],waves[2015]
    unchanged2 = code_number(w2["bd001_w2_4"]).eq(12)
    previous2 = numeric_subset(w2["bd001_w2_3"],range(1,12)).combine_first(
        numeric_subset(w2["zbd001"],range(1,12)))
    current2 = numeric_subset(w2["bd001_w2_4"],range(1,12)).combine_first(
        previous2.where(unchanged2)).combine_first(numeric_subset(w2["bd001"],range(1,12)))
    w2["education"] = education_levels(current2)
    # Use the 2011 report only if a 2013 interview is absent and 2015 says unchanged.
    w1_old = numeric_subset(waves[2011]["bd001"],range(1,12)).reindex(w3.index)
    history = current2.reindex(w3.index).combine_first(w1_old.where(
        ~w2["in_demo"].reindex(w3.index).eq(True)))
    current3 = numeric_subset(w3["bd001_w2_4"],range(1,12)).combine_first(
        history.where(code_number(w3["bd001_w2_4"]).eq(12)))
    w3["education"] = education_levels(current3)


