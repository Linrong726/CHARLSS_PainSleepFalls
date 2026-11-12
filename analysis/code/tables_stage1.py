"""Build candidate manuscript tables, trace numeric cells and stage workbook data."""
import csv
import json
from pathlib import Path
import sys

GROUPS = ("G1", "G2", "G3", "G4")
DISPLAY = ("Pain yes, restless sleep frequent", "Pain yes, restless sleep not frequent",
           "Pain no, restless sleep frequent", "Pain no, restless sleep not frequent")
T1 = [
    ("Age, years", "age", None), ("Female", "sex", 2),
    ("Education, no school and illiterate", "education_R", 0),
    ("Education, below lower secondary*", "education_R", 1),
    ("Education, lower secondary or higher", "education_R", 2), ("Rural residence", "rural", 1),
    ("Reported a fall in prior 24 months", "prior_fall", 1),
    ("Restless sleep, 3–4 days/week", "sleep_severity_2011", 0),
    ("Restless sleep, 5–7 days/week", "sleep_severity_2011", 1),
    ("Mood symptom score, 0–27", "mood9", None), ("    Missing", "mood9", "missing"),
    ("Harmonized chronic-condition count, 0–12", "chronic12", None), ("    Missing", "chronic12", "missing"),
    ("Difficulty walking 100 m", "walk100", 1), ("    Missing", "walk100", "missing"),
    *[(f"Distance vision, level {level}", "vision", level) for level in range(1, 6)],
    *[(f"Self-rated memory, level {level}", "memory", level) for level in range(1, 6)],
]


def read(root, name):
    with (root / "results" / f"{name}.csv").open(encoding="utf-8-sig", newline="") as file:
        return [dict(row, _row=index) for index, row in enumerate(csv.DictReader(file), 2)]


def lookup(rows, **match):
    return next(row for row in rows if all(row[key] == value for key, value in match.items()))


def number(row, key, digits=0):
    return f"{float(row[key]):,.{digits}f}"


def rr(row):
    return f"{number(row, 'estimate', 3)} ({number(row, 'ci_low', 3)}–{number(row, 'ci_high', 3)})"


def write(root, name, rows):
    with (root / "results" / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as file:
        csv.writer(file).writerows(rows)


def build(root):
    master, groups, baseline, flow, selection = (
        read(root, name) for name in ("results_master", "sample_groups", "baseline", "flow", "selection"))
    t1 = [["Baseline characteristic", *[f"{gid}  n = {lookup(groups, sample_id='P', group_id=gid)['n']}"
                                        for gid in GROUPS]]]
    index = [["location", "source_csv", "source_rows", "source_fields", "rendering"]]

    def link(location, source, row, fields, formatting):
        index.append([location, source + ".csv", row["_row"], fields, formatting])

    for gid in GROUPS:
        link(f"Table 1 {gid} header", "sample_groups", lookup(groups, sample_id="P", group_id=gid), "n", "integer")
    for i, (label, field, level) in enumerate(T1, 1):
        row = [label]
        key = "" if level is None or (level == "missing" and field != "walk100") else (
            "1.0" if level == "missing" else f"{level:.1f}")
        for group in GROUPS:
            data = lookup(baseline, sample="B_M1_R", group_id=group, field=field, level=key)
            if level == "missing":
                fields = "missing"
                value = data["missing"]
            elif level is None:
                fields = "mean|sd|n"
                value = f"{number(data, 'mean', 2)} ({number(data, 'sd', 2)})\nn = {data['n']}"
            else:
                fields = "count|n|proportion"
                value = f"{number(data, 'count')}/{data['n']} ({float(data['proportion']):.1%})"
            row.append(value)
            link(f"Table 1 row {i} {group}", "baseline", data, fields, "as printed in table_1.csv")
        t1.append(row)
    t2 = [["2015 symptom reports", "Valid outcome", "Falls", "Crude %", "Adjusted RR (95% CI)", "Role"]]
    for gid, label in zip(GROUPS, DISPLAY):
        grp = lookup(groups, sample_id="P", group_id=gid)
        link(f"Table 2 {gid}", "sample_groups", grp, "n|events|crude_proportion", "as printed in table_2.csv")
        rr_label = "1.000 (reference)"
        if gid != "G1":
            row = lookup(master, model_id="MAIN", group_id=gid)
            rr_label = rr(row)
            link(f"Table 2 {gid}", "results_master", row, "estimate|ci_low|ci_high", "3 decimals")
        t2.append([f"{gid}  {label}", grp["n"], grp["events"], f"{float(grp['crude_proportion']):.1%}",
                   rr_label, "Reference" if gid == "G1" else "Primary" if gid == "G4" else "Secondary"])
    t3 = [["Selection stage", "Retained / eligible", "Excluded", "Selected baseline contrast", "SMD"]]
    stages = (
        ("2015 classification", "B_START", "B_CLASS", "CLASS_vs_UNCLASS", "age", "", "Age (years), mean"),
        ("2018 outcome availability", "B_CLASS", "P", "P_vs_NO_VALID_Y", "age", "", "Age (years), mean"),
        ("Full health complete case", "P", "F", "F_vs_P_NOT_F", "prior_fall", "1.0", "Prior fall, count/valid n"),
        ("Walk requirement relaxed", "P", "R0", "R0_vs_P_NOT_R0", "mood9", "", "Mood score, mean"),
        ("Original 2011 count valid", "R0", "RSTAR", None, None, None, "All R0 retained"),
    )
    for label, denominator, kept, comparison, field, level, desc in stages:
        before, after = (lookup(flow, stage=tag) for tag in (denominator, kept))
        for row in (before, after):
            link(f"Table 3 {label}", "flow", row, "n", "retained / eligible; difference excluded")
        smd = "—"
        if comparison:
            cmp = lookup(selection, comparison=comparison, field=field, level=level)
            smd = number(cmp, "smd", 2)
            if level:
                desc += f" {number(cmp, 'retained_count')}/{cmp['retained_n']} vs {number(cmp, 'excluded_count')}/{cmp['excluded_n']}"
            else:
                desc += f" {number(cmp, 'retained_mean', 2)} vs {number(cmp, 'excluded_mean', 2)}"
            link(f"Table 3 {label}", "selection", cmp, "smd|retained/excluded mean or count and n",
                 "as printed in table_3.csv")
        t3.append([label, f"{number(after,'n')}/{number(before,'n')}",
                   f"{int(before['n']) - int(after['n']):,}", desc, smd])
    for i, table in enumerate((t1, t2, t3), 1):
        write(root, f"table_{i}", table)
    def block(location, name, predicate, fields, formatting):
        for row in read(root, name):
            if predicate(row):
                link(location, name, row, fields, formatting)
    block("Abstract, Results and Figure 4", "results_master",
          lambda r: r["estimand"] == "RR" and r["contrast_id"] == "G4_vs_G1",
          "n|events|clusters|estimate|ci_low|ci_high", "RR and CI at 3 decimals")
    block("Primary Results", "results_master", lambda r: r["model_id"] == "MAIN",
          "estimate|ci_low|ci_high", "RR and CI at 3 decimals")
    block("Abstract, Results and Figure 3", "results_master", lambda r: r["estimand"] in ("PROB", "RD"),
          "estimate|ci_low|ci_high", "multiply by 100; 1 decimal")
    block("Results, Methods and Figure 1", "flow", lambda r: True, "n|events|clusters", "stage counts")
    block("Cohort Results and Figure 2", "followup", lambda r: True, "n|denominator|percent", "counts and percentages")
    block("Cohort Results", "followup_detail", lambda r: True, "n", "sum across groups by reason")
    block("Cohort Results", "selection",
          lambda r: r["comparison"] == "P_vs_NO_VALID_Y" and r["field"] == "age",
          "retained_mean|excluded_mean|smd", "2 decimals for means, 3 for SMD")
    block("Cohort Results and recall window", "month_summary", lambda r: r["group_id"] == "ALL",
          "min_months|median_months|max_months", "integer calendar months")
    block("S3 Results", "s3", lambda r: True, "n|e|m|denominator|lower|upper", "counts; proportions as %")
    for name, fields in (("missing", "n_missing|denominator"), ("walk_source", "db001|db002|db003|n"),
                         ("chronic", "n|denominator"), ("probability_scenarios", "n|minimum|maximum|outside_tolerance")):
        block("Source, Stage 1 or probability Results", name, lambda r: True, fields, "as cited in paragraph")
    for location, name in (
        ("Additional file 1 S3", "selection_exclusion_reasons"), ("Additional file 1 S11", "followup_detail"),
        ("Additional file 1 S2", "month_summary"), ("Additional file 1 S4", "s3"),
        ("Additional file 1 selection tables", "selection"), ("Additional file 1 source tables", "walk_source"),
        ("Additional file 1 source tables", "chronic"), ("Additional file 1 model tables", "results_master"),
        ("Additional file 1 probability diagnostics", "probability_scenarios")):
        block(location, name, lambda r: True, "reported subset of columns", "see table heading and source labels")
    fig = root / "figures" / "FIGURE_SOURCE_LONG.csv"
    if fig.exists():
        with fig.open(encoding="utf-8-sig", newline="") as file:
            for i, row in enumerate(csv.DictReader(file), 2):
                index.append([row["figure_id"], "../figures/FIGURE_SOURCE_LONG.csv", i,
                              "n|events|estimate|CI", "see long-table panel and underlying CSV source"])
    write(root, "numeric_index", index)
    workbook_tabs = (
        ("Results", "results_master"), ("Flow", "flow"), ("Followup", "followup"),
        ("GroupCounts", "sample_groups"), ("Selection", "selection"), ("Missing", "missing"),
        ("MissingPatterns", "missing_patterns"), ("WalkSource", "walk_source"),
        ("ChronicSource", "chronic"), ("ModelSpecs", "model_specifications"),
        ("ModelDiagnostics", "model_diagnostics"), ("Scenarios", "probability_scenarios"),
        ("ProbabilityCovariance", "probability_covariance"), ("NumericIndex", "numeric_index"),
    )
    def value(s):
        if s == "":
            return None
        try:
            return int(s) if s.lstrip("-").isdigit() else float(s)
        except ValueError:
            return s
    specs = []
    for name, filename in workbook_tabs:
        with (root / "results" / f"{filename}.csv").open(encoding="utf-8-sig", newline="") as file:
            head, *data = list(csv.reader(file))
        specs.append({"name": name, "values": [head, *[[value(s) for s in row] for row in data]]})
    with (Path(__file__).parent / "reference" / "v211" / "dictionary_csv" / "01_核心变量.csv").open(
            encoding="utf-8-sig", newline="") as file:
        specs.append({"name": "Dictionary", "values": list(csv.reader(file))})
    (root / "run" / "workbook_data.json").write_text(json.dumps(specs, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    build(Path(sys.argv[1]))
