
"""W1 reconstruction using the unchanged v1.1 reader and outcome rules."""
import argparse
import json
from pathlib import Path

from audit_core import (Inputs, M1, M2, GROUPS, code_number, exact_index, join,
                        load_wave, outcome_status, strict_order, string_codes)

WAVES = (2011, 2015, 2018)
EXPOSURE = ["p","s","proxy","proxy_unknown","date_lo","date_hi","pain_code","sleep_code",
            "in_demo","in_health","in_sleep_module"]
OUTCOME = ["in_demo","in_health","date_lo","date_hi","dead","proxy","proxy_unknown",
           "route","last_lo","last_hi","new_fall","fall","fall_code","treated_count"]


def state(frame, suffix):
    return (frame[f"p{suffix}"].map({0:"0",1:"1"}) +
            frame[f"s{suffix}"].map({0:"0",1:"1"}))


def build_cohort(raw_root: Path):
    inp = Inputs(raw_root)
    harmonized = inp.harmonized()
    bridge_frame = harmonized.loc[harmonized.ID_w1.ne(""), ["ID_w1"]].reset_index()
    bridge = exact_index(bridge_frame,"ID_w1","Harmonized D baseline bridge",inp.qc)["ID"]
    assert bridge.index.str.fullmatch(r"[0-9]{11}").all() and bridge.is_unique
    # Verify official format without using a constructed ID for any join.
    agreement = int((bridge.index.str[:9]+"0"+bridge.index.str[9:] == bridge.to_numpy()).sum())
    assert agreement == len(bridge)
    inp.qc.append(dict(stage="Bridge compared to official 2013 release-note rule",
                       rows=len(bridge),overlap=agreement,status="PASS"))
    w0,w1,w2 = (load_wave(inp,w,harmonized,bridge) for w in WAVES)
    d = w0.loc[w0.in_demo.eq(True), EXPOSURE+M2+["cluster","bd001"]].copy()
    d["bd001"] = string_codes(d["bd001"])
    d = d.rename(columns={name:name+"0" for name in EXPOSURE})
    d = join(d,w1[EXPOSURE+["dead","iwstat"]].add_suffix("1"),"W1 join t1",inp.qc)
    d = join(d,w2[OUTCOME].add_suffix("2"),"W1 join t2",inp.qc)
    d["iwstat2"] = code_number(harmonized.r4iwstat.reindex(d.index))
    death = d.iwstat2.isin([5,6]) & ~d.in_demo2.eq(True)
    d.loc[death,"dead2"] = 1.
    d["state0"], d["state1"] = state(d,0), state(d,1)
    d["Bgroup"] = d.state1.map(GROUPS)
    d["Agroup"] = d.s0.map({0:"0",1:"1"}) + d.s1.map({0:"0",1:"1"})
    d["outcome_status"], d["Y"] = outcome_status(d,True)
    d["M1_complete"] = d[M1].notna().all(axis=1)
    d["M2_complete"] = d[M2].notna().all(axis=1)
    d["B_start"] = d.age.ge(60) & d.state0.eq("11")
    d["B_class"] = d.B_start & d.in_demo1.eq(True) & d.state1.notna()
    d["A_class"] = (d.age.ge(60) & d.in_demo1.eq(True) & d.state0.notna() &
                    d.state1.notna() & d.p0.eq(1) & d.p1.eq(1))
    d["t0_before_t1"] = strict_order(d.date_hi0,d.date_lo1)
    d["t1_before_t2"] = strict_order(d.date_hi1,d.date_lo2)
    d["exact_t1_anchor"] = d.last_lo2.notna() & d.last_lo2.eq(d.date_lo1)
    b_valid = d.B_class & d.Y.notna()
    assert d.index.is_unique
    assert d.loc[b_valid,["in_demo1","in_demo2","t0_before_t1",
                          "t1_before_t2","exact_t1_anchor"]].all().all()
    assert d.loc[b_valid,"route2"].eq(1).all()
    assert not d.loc[b_valid,"new_fall2"].notna().any()
    invalid = d.dead2.eq(1) | ~d.in_health2.eq(True) | d.proxy2.eq(True)
    assert d.loc[invalid,"Y"].isna().all()

    health = w2.loc[w2.in_health.eq(True)]
    returned = health.route.eq(1)
    new = health.route.eq(2)
    wrong = (returned & health.new_fall.notna()) | (new & health.fall.notna())
    route_audit = dict(n=len(health), return_n=int(returned.sum()), new_n=int(new.sum()),
        return_binary=int((returned & health.fall.notna()).sum()),
        new_binary=int((new & health.new_fall.notna()).sum()),
        wrong_route=int(wrong.sum()),
        both_routes=int((health.fall.notna() & health.new_fall.notna()).sum()),
        last_month_on_return=int((returned & health.last_lo.notna()).sum()))
    assert route_audit["wrong_route"] == route_audit["both_routes"] == 0
    return d, {
        "inputs": inp,
        "bridge_agreement": agreement,
        "route_audit": route_audit,
        "wave_demographics": {str(w):int(frame.in_demo.eq(True).sum())
                              for w,frame in zip(WAVES,(w0,w1,w2))},
        "bd001_metadata": inp.metadata[(2011,"demographic_background","bd001")],
        "fall_code_metadata": inp.metadata[(2018,"health_status_and_functioning","da023_w4")],
    }


def count_events(frame):
    assert frame.Y.isin([0,1]).all()
    return dict(N=len(frame),events=int(frame.Y.eq(1).sum()),
                non_events=int(frame.Y.eq(0).sum()))


def reconstruction(d):
    obs = d.loc[d.B_class & d.Y.notna()]
    m1 = obs.loc[obs.M1_complete & obs.cluster.notna()]
    m2 = obs.loc[obs.M2_complete & obs.cluster.notna()]
    history = d.loc[d.A_class & d.Y.notna() & d.M1_complete & d.cluster.notna()]
    return dict(
        age60=int(d.age.ge(60).sum()),
        t0_double_positive=int(d.B_start.sum()),
        t1_classified=int(d.B_class.sum()),
        B_valid_Y=count_events(obs),
        B_M1_old=count_events(m1),
        B_M2_old=count_events(m2),
        B_group_counts={k:count_events(obs.loc[obs.Bgroup.eq(k)]) for k in GROUPS.values()},
        history_A=count_events(history),
        history_A_clusters=int(history.cluster.nunique()),
        history_A_groups={k:count_events(history.loc[history.Agroup.eq(k)])
                          for k in ("11","10","01","00")},
        Y_any_t1_class_from_B_start=int((d.B_start & d.Y.notna()).sum()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root",type=Path)
    args = parser.parse_args()
    d, ev = build_cohort(args.raw_root)
    files = ev.pop("inputs").verify()
    assert all(r["unchanged"] for r in files)
    print(json.dumps({"cohort":reconstruction(d),"evidence":ev,
                      "sources_unchanged":len(files)},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()


