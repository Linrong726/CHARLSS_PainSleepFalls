"""Render publication-scale CHARLS figures from frozen Stage 1 v1.1 aggregate results."""

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
import math

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import MultipleLocator, PercentFormatter
from matplotlib.transforms import blended_transform_factory

FULL_WIDTH = 170 / 25.4

INK = "#202C37"
MUTED = "#465767"
GRID = "#DDE4E9"
BLUE = "#215979"
BLUE_LIGHT = "#EAF3F7"
ORANGE = "#B96537"
PURPLE = "#6D597A"
GROUPS = {
    "G1": ("Pain reported", "Frequent restless sleep"),
    "G2": ("Pain reported", "Restless sleep below threshold"),
    "G3": ("Pain item negative", "Frequent restless sleep"),
    "G4": ("Pain item negative", "Restless sleep below threshold"),
}
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 11,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": MUTED,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.linewidth": 0.65,
}
LONG_FIELDS = (
    "figure_id", "panel", "sample_id", "model_id", "group_id",
    "contrast_id", "standardization_reference", "status_or_outcome",
    "level", "n", "events", "clusters", "denominator", "estimate",
    "ci_low", "ci_high", "unit", "source_table", "source_row",
    "source_column", "role",
)
MAP_FIELDS = (
    "item_id", "placement", "question", "sample_and_denominator",
    "source", "identity", "legacy_mapping",
)


def read_table(directory, filename):
    with (directory / filename).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {**row, "_row": number, "_table": filename}
            for number, row in enumerate(reader, start=2)
        ]


def select(rows, **where):
    return [row for row in rows if all(row.get(key) == value for key, value in where.items())]


def one(rows, **where):
    found = select(rows, **where)
    if len(found) != 1:
        raise ValueError(f"Expected one source row for {where}; found {len(found)}")
    return found[0]


def num(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"{row['_table']} row {row['_row']} column {key} is not finite")
    return value


def optional_num(row, key):
    value = row.get(key, "")
    if value is None or value.strip().upper() in {"", "NA", "NAN", "NOT_ESTIMABLE", "NOT_COMPUTABLE"}:
        return None
    return num(row, key)


def count(row, key="n"):
    value = num(row, key)
    if value < 0 or not value.is_integer():
        raise ValueError(f"{row['_table']} row {row['_row']} column {key} is not a count")
    return int(value)


def estimate_interval(row):
    est, lo, hi = (num(row, key) for key in ("estimate", "ci_low", "ci_high"))
    if not lo <= est <= hi:
        raise ValueError(f"Confidence interval does not contain its estimate: {row}")
    return est, lo, hi


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def dot_interval(ax, y, estimate, lo, hi, color=BLUE, marker="o"):
    point, _, (interval,) = ax.errorbar(
        estimate, y, xerr=[[estimate - lo], [hi - estimate]],
        fmt=marker, color=color, lw=1.35, ms=5.2, capsize=2.2, zorder=3,
    ).lines
    (left, right), = interval.get_segments()
    assert math.isclose(point.get_xdata()[0], estimate)
    assert math.isclose(left[0], lo) and math.isclose(right[0], hi)


@dataclass
class Release:
    directory: Path
    long_rows: list = field(default_factory=list)
    legend_blocks: list = field(default_factory=list)
    item_rows: list = field(default_factory=list)
    current_id: str = ""

    def track(self, row, *, panel="", status="", unit="", source_column="",
              estimate_key="estimate", n_key="n", **override):
        values = {
            "figure_id": self.current_id,
            "panel": panel,
            "sample_id": row.get("sample_id", ""),
            "model_id": row.get("model_id", ""),
            "group_id": row.get("group_id", ""),
            "contrast_id": row.get("contrast_id", ""),
            "standardization_reference": row.get("standardization_reference", ""),
            "status_or_outcome": status,
            "level": row.get("level", ""),
            "n": row.get(n_key, ""),
            "events": row.get("events", ""),
            "clusters": row.get("clusters", ""),
            "denominator": row.get("denominator", ""),
            "estimate": row.get(estimate_key, ""),
            "ci_low": row.get("ci_low", ""),
            "ci_high": row.get("ci_high", ""),
            "unit": unit,
            "source_table": row["_table"],
            "source_row": row["_row"],
            "source_column": source_column,
            "role": row.get("role", ""),
        }
        self.long_rows.append({**values, **override})

    def register(self, item_id, placement, question, sample, source, identity, legacy):
        self.item_rows.append(dict(zip(MAP_FIELDS, (
            item_id, placement, question, sample, source, identity, legacy
        ))))

    def save(self, fig, title, legend, alt, **mapping):
        fig.canvas.draw()
        prefix = self.directory / self.current_id
        fig.savefig(
            prefix.with_suffix(".pdf"),
            metadata={"Title": title, "CreationDate": None, "ModDate": None},
        )
        fig.savefig(prefix.with_suffix(".png"), dpi=320)
        plt.close(fig)
        self.legend_blocks.append(
            f"## {title}\n\n{legend}\n\nAlt text. {alt}\n\n"
            f"Suggested placement. {mapping['placement']}.\n"
        )
        self.register(self.current_id, **mapping)

    def finish(self):
        write_csv(self.directory / "FIGURE_SOURCE_LONG.csv", self.long_rows, LONG_FIELDS)
        write_csv(self.directory / "DISPLAY_ITEM_MAP.csv", self.item_rows, MAP_FIELDS)
        (self.directory / "FIGURE_LEGENDS_AND_ALT_TEXT.md").write_text(
            "# Figure legends and alternative text\n\n"
            "All counts and estimates come from frozen aggregate CSVs. Source row numbers "
            "count the CSV header as row 1. Probability and RD values in the long table "
            "use a 0–1 scale. The figures show percentages or percentage points.\n\n"
            + "\n".join(self.legend_blocks), encoding="utf-8", newline="\n"
        )


def box(ax, x, y, width, height, line, number, accent=False):
    artist = FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.006,rounding_size=0.013",
        fc=BLUE_LIGHT if accent else "white", ec=BLUE if accent else "#AAB8C2",
        lw=1.0 if accent else 0.7, zorder=3,
    )
    ax.add_patch(artist)
    ax.text(x + width / 2, y + height / 2, f"{line}  (n = {number:,})",
            ha="center", va="center", zorder=4, fontsize=8.3)


def arrow(ax, start, end, color=MUTED, style="-|>"):
    ax.annotate("", xy=end, xytext=start, zorder=2, arrowprops={
        "arrowstyle": style, "color": color, "lw": 0.85,
        "mutation_scale": 8, "shrinkA": 0, "shrinkB": 1.5,
    })


def figure1(release, flow):
    release.current_id = "FIGURE_1"
    stages = ("AGE60", "B_START", "B_CLASS", "UNCLASSIFIED",
              "NO_VALID_Y", "P", "R0", "RSTAR", "F")
    records = {stage: one(flow, stage=stage) for stage in stages}
    numbers = {stage: count(row) for stage, row in records.items()}
    if (
        numbers["B_CLASS"] + numbers["UNCLASSIFIED"] != numbers["B_START"]
        or numbers["P"] + numbers["NO_VALID_Y"] != numbers["B_CLASS"]
        or not numbers["F"] <= numbers["RSTAR"] <= numbers["R0"] <= numbers["P"]
    ):
        raise ValueError("Cohort counts do not close for Figure 1.")
    for stage, row in records.items():
        release.track(row, panel="B", sample_id=stage, status=stage,
                      source_column="n|events|clusters", role="cohort accounting")
    fig = plt.figure(figsize=(FULL_WIDTH, 5.28))
    timeline = fig.add_axes([0.035, 0.685, 0.93, 0.295], frameon=False)
    flow_ax = fig.add_axes([0.035, 0.02, 0.93, 0.62], frameon=False)
    for ax in (timeline, flow_ax):
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.set_axis_off()
    timeline.text(0, 0.97, "A   Interviews and fall recall window", fontsize=10.5,
                  fontweight="bold", va="top")
    for x, year, label in (
        (0.15, "2011", "Age ≥60 years\nPain and frequent restless sleep"),
        (0.50, "2015", "Repeat both symptom items\nClassify four report groups"),
        (0.85, "2018", "Any fall since the verified\n2015 interview"),
    ):
        timeline.text(x, 0.65, year, ha="center", va="center",
                      fontsize=11, fontweight="bold", color=BLUE)
        timeline.plot(x, 0.50, "o", ms=5, color=BLUE, zorder=4)
        timeline.text(x, 0.38, label, ha="center", va="top", fontsize=8.5)
    arrow(timeline, (0.15, 0.50), (0.86, 0.50))
    timeline.plot([0.50, 0.50, 0.85, 0.85], [0.16, 0.12, 0.12, 0.16], color=BLUE, lw=1)
    timeline.text(0.675, 0.06, "Participant-specific recall window",
                  ha="center", va="center", fontsize=8.2, color=BLUE)
    flow_ax.text(0, 0.97, "B   Selection and nested analysis samples", fontsize=10.5,
                 fontweight="bold", va="top")
    main_y = [0.77, 0.59, 0.41, 0.23]
    main_x, main_width, main_h = 0.02, 0.48, 0.105
    main_center = main_x + main_width / 2
    for i, (stage, label) in enumerate((
        ("AGE60", "2011 interview, age ≥60 years"),
        ("B_START", "Both baseline symptoms reported"),
        ("B_CLASS", "2015 reports classifiable"),
        ("P", "P: Valid outcome and base covariates"),
    )):
        box(flow_ax, main_x, main_y[i], main_width, main_h,
            label, numbers[stage], accent=stage == "P")
        if i:
            arrow(flow_ax, (main_center, main_y[i - 1]), (main_center, main_y[i] + main_h))
    for i, (stage, label) in enumerate((
        ("UNCLASSIFIED", "2015 reports unclassifiable"),
        ("NO_VALID_Y", "No valid 2018 outcome"),
    )):
        y = [0.51, 0.33][i]
        start_y = y + main_h / 2
        box(flow_ax, 0.59, y, 0.39, main_h, label, numbers[stage])
        arrow(flow_ax, (main_center, start_y), (0.59, start_y))
    flow_ax.text(0.02, 0.113, "Within P", color=BLUE, fontsize=8.1,
                 va="center", fontweight="bold")
    for i, (stage, name) in enumerate((("R0", "R0"), ("RSTAR", "R*"), ("F", "F"))):
        x = 0.15 + 0.285 * i
        box(flow_ax, x, 0.059, 0.24, 0.105, name, numbers[stage], accent=True)
        if i:
            arrow(flow_ax, (x - 0.045, 0.111), (x, 0.111), color=BLUE)
    arrow(flow_ax, (main_center, 0.23), (main_center, 0.164), color=BLUE)
    release.save(
        fig, "Figure 1. Interview timing and cohort construction",
        "Panel A shows the interviews used to assess baseline eligibility, repeat symptom reports "
        "and ascertain falls. The outcome is any fall over the verified participant-specific recall "
        "window. Panel B shows the cohort selection and analysis samples. P is the primary sample. "
        "R0 retains all extended complete-case criteria apart from observed 100 m walking "
        "difficulty. R* also requires an available original 2011 condition count. F meets the "
        "original extended complete-case criteria. The samples are nested as F ⊆ R* ⊆ R0 ⊆ P.",
        "Timeline of the 2011, 2015 and 2018 interviews above a cohort flow chart that ends "
        f"in P with {numbers['P']:,} participants. R0, R* and F appear as progressively nested subsets.",
        placement="Main text, study population",
        question="Who entered the analysis and when were exposures and outcomes measured?",
        sample="2011 age-eligible interviewees through P, R0, R* and F",
        source="flow.csv", identity="Expanded baseline figure", legacy="Old Figure 1",
    )


def figure2(release, followup):
    release.current_id = "FIGURE_2"
    statuses = (
        ("FALL", "Valid fall report", BLUE, None, "white"),
        ("NO_FALL", "Valid no-fall report", "#A7CAD9", None, INK),
        ("DEATH", "Verified death", "#626A73", "////", "white"),
        ("OTHER", "Other unavailable", "#F3EDE0", "...", INK),
    )
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.38))
    fig.subplots_adjust(left=0.36, right=0.96, top=0.98, bottom=0.22)
    labels = []
    total = 0
    for y, (group, label) in enumerate(GROUPS.items()):
        rows = [one(followup, group_id=group, status=status) for status, *_ in statuses]
        denominator = count(rows[0], "denominator")
        if any(count(row, "denominator") != denominator for row in rows):
            raise ValueError(f"Inconsistent follow-up denominator for {group}")
        if sum(count(row) for row in rows) != denominator:
            raise ValueError(f"Follow-up records do not close for {group}")
        pain, sleep = label
        labels.append(f"{group}  {pain} (n = {denominator:,})\n{sleep}")
        total += denominator
        left = 0
        for row, (status, _, color, hatch, text_color) in zip(rows, statuses):
            n = count(row)
            percent = 100 * n / denominator
            if not math.isclose(num(row, "percent"), percent, abs_tol=0.005):
                raise ValueError(f"Follow-up percent differs from n/denominator: {row}")
            bar, = ax.barh(y, percent, left=left, height=0.61, color=color,
                            edgecolor=INK if hatch else "white", linewidth=0.6, hatch=hatch)
            assert math.isclose(bar.get_x(), left) and math.isclose(bar.get_width(), percent)
            if percent >= 5.2:
                ax.text(
                    left + percent / 2, y, f"{n:,}", color=text_color,
                    ha="center", va="center", fontsize=8.0, fontweight="bold",
                    bbox={"facecolor": color, "edgecolor": "none", "pad": 0.65} if hatch else None,
                )
            release.track(
                row, panel="A", sample_id="B_CLASS", status=status, unit="percent",
                estimate_key="percent", source_column="n|denominator|percent",
                role="follow-up availability", events=n if status == "FALL" else "",
            )
            left += percent
    ax.set(yticks=range(len(GROUPS)), yticklabels=labels, xlim=(0, 100))
    ax.invert_yaxis()
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=9)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_major_formatter(PercentFormatter(100))
    ax.set_xlabel("Share of each 2015 report group")
    fig.legend(
        handles=[Patch(facecolor=color, hatch=hatch, edgecolor=MUTED, label=label)
                 for _, label, color, hatch, _ in statuses],
        loc="lower center", bbox_to_anchor=(0.5, 0.015), frameon=False,
        ncol=4, fontsize=8.2,
    )
    release.save(
        fig, "Figure 2. Follow-up record availability",
        f"Bars show mutually exclusive 2018 record categories within the four classifiable "
        f"2015 report groups (total n = {total:,}). The sleep criterion was restless sleep on at "
        "least 3 days in the previous week. Valid fall and no-fall reports met the outcome "
        "ascertainment rules. Verified death and other unavailable outcomes indicate why valid "
        "outcomes were unavailable. Death does not establish whether a participant fell before "
        "dying. These percentages describe outcome availability. Labels give record counts. "
        "Table S11 in Additional file 1 details all records. Reported pain denotes an affirmative "
        "response to being often troubled by bodily pain. Below threshold denotes restless sleep "
        "on fewer than 3 days in the previous week.",
        "Four stacked bars show valid fall reports, valid no-fall reports, verified deaths and "
        "other unavailable outcomes within each classifiable 2015 group.",
        placement="Main text, cohort retention and availability",
        question="How does outcome availability vary across classifiable 2015 report groups?",
        sample=f"B_CLASS, n={total}; group-specific denominator",
        source="followup.csv", identity="New descriptive analysis", legacy="Extends old Table S3",
    )


def signed(value):
    return f"{value:+.1f}".replace("-", "−")


def figure3(release, results):
    rows = [r for r in results if r["estimand"].upper() in {"PROB", "RD"}]
    release.current_id = "FIGURE_3"
    models = {r["model_id"] for r in rows}
    if models == {"ABS_POISSON"}:
        model_text = "Modified Poisson"
    elif models == {"ABS_LOGISTIC"}:
        model_text = "Auxiliary logistic"
    else:
        raise ValueError(f"Expected one probability model, found {models}")
    probabilities = select(rows, estimand="PROB")
    rd = one(rows, estimand="RD")
    n = count(rd)
    references = {row["standardization_reference"] for row in rows}
    if len(references) != 1 or not next(iter(references)) or len(rows) != 5:
        raise ValueError("Figure 3 needs all four probabilities, one RD, and a common reference")
    rd_est, rd_lo, rd_hi = (100 * x for x in estimate_interval(rd))
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.14))
    fig.subplots_adjust(left=0.36, right=0.705, top=0.91, bottom=0.245)
    fig.text(0.745, 0.98, "Probability (95% CI)", fontsize=8.4,
             va="top", fontweight="bold")
    fig.text(0.02, 0.98, f"Primary sample (n = {n:,})", fontsize=8.4,
             va="top", fontweight="bold")
    transform = blended_transform_factory(fig.transFigure, ax.transData)
    upper = 0
    plotted = {}
    for y, (group, label) in enumerate(GROUPS.items()):
        row = one(probabilities, group_id=group)
        if count(row) != n:
            raise ValueError(f"Inconsistent standardization N for {group}")
        estimate, lo, hi = (100 * value for value in estimate_interval(row))
        if not 0 <= lo <= hi <= 100:
            raise ValueError(f"Probability interval outside 0–100% for {group}")
        plotted[group] = estimate
        primary = group in ("G1", "G4")
        dot_interval(ax, y, estimate, lo, hi, BLUE if primary else MUTED,
                     "o" if primary else "s")
        pain, sleep = label
        ax.text(0.02, y, f"{group}  {pain}\n{sleep}", ha="left", va="center",
                transform=transform, color=BLUE if primary else MUTED,
                fontweight="bold" if primary else "normal", fontsize=8.5)
        ax.text(0.745, y, f"{estimate:.1f}% ({lo:.1f}%–{hi:.1f}%)",
                ha="left", va="center", transform=transform, fontsize=8.6)
        upper = max(upper, hi)
        release.track(row, panel="A", status="fall report probability", unit="proportion",
                      source_column="estimate|ci_low|ci_high|n|events|clusters")
    if not math.isclose(plotted["G4"] - plotted["G1"], rd_est, abs_tol=1e-8):
        raise ValueError("RD differs from G4 minus G1.")
    release.track(rd, panel="A", status="G4 minus G1", unit="proportion difference",
                  source_column="estimate|ci_low|ci_high|n|events|clusters")
    ax.set(yticks=[], xlim=(0, min(100, max(10, math.ceil(upper / 10) * 10))),
           ylim=(3.55, -0.55))
    ax.set_xlabel("Probability of reporting a fall")
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(PercentFormatter(100))
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.spines["left"].set_visible(False)
    fig.text(
        0.02, 0.045,
        f"G4 minus G1: {signed(rd_est)} percentage points (95% CI {signed(rd_lo)} to {signed(rd_hi)})",
        fontsize=9.1, color=BLUE, fontweight="bold",
    )
    release.save(
        fig, "Figure 3. Standardized fall-report probabilities",
        f"Points are average predicted probabilities from the {model_text[0].lower() + model_text[1:]} model over "
        "the same fixed empirical covariate distribution in P. Error bars are nominal 95% "
        "confidence intervals based on community-clustered variance and the delta method. "
        "The G4 minus G1 difference uses the covariance between predictions. G2 and G3 are "
        "secondary descriptions. The probabilities summarize adjusted associations in P. Pain "
        "denotes an affirmative response to being often troubled by bodily pain. The frequent "
        "restless sleep threshold is at least 3 days in the previous week.",
        f"Four dots with intervals show standardized fall-report probabilities in P. "
        f"The annotated G4 minus G1 difference is {signed(rd_est)} percentage points.",
        placement="Main text, standardized probabilities",
        question="What absolute fall-report probabilities correspond to the adjusted comparison in P?",
        sample=f"P, n={n}; the same empirical covariate distribution for all four groups",
        source="results_master.csv", identity="New post hoc descriptive analysis", legacy="New figure",
    )


FOREST = (
    ("Primary and prior sensitivity models", (
        ("MAIN", "Base in P"),
        ("S1", "Base without prior falls"),
        ("S4", "Base with recall months"),
    )),
    ("Complete-case model bridge", (
        ("F_BRIDGE", "Base in F"),
        ("F_HARM", "Full, Harmonized count"),
        ("F_RAW", "Full, original 2011 count"),
    )),
    ("Post hoc targeted sensitivity models", (
        ("N1", "F_RAW without walking"),
        ("N2", "N1 formula in R*"),
        ("N3", "Base in R*"),
    )),
)


def figure4(release, results):
    release.current_id = "FIGURE_4"
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 4.82))
    fig.subplots_adjust(left=0.49, right=0.735, top=0.98, bottom=0.115)
    transform = blended_transform_factory(fig.transFigure, ax.transData)
    col = (0.02, 0.455, 0.765)
    for x, label, ha in zip(col, ("Model", "N / reports", "RR (95% CI)"),
                            ("left", "right", "left")):
        ax.text(x, -0.67, label, transform=transform, ha=ha, va="center",
                fontsize=8.3, fontweight="bold")
    y = 0.4
    lowers, uppers = [], []
    for title, models in FOREST:
        ax.text(col[0], y, title, transform=transform, va="center", ha="left",
                fontsize=8.7, fontweight="bold", color=BLUE)
        y += 0.8
        for model_id, label in models:
            row = one(results, model_id=model_id, contrast_id="G4_vs_G1")
            estimate, lo, hi = estimate_interval(row)
            if lo <= 0:
                raise ValueError(f"Non-positive interval limit for log-scale RR: {row}")
            lowers.append(lo)
            uppers.append(hi)
            color = BLUE if model_id == "MAIN" else MUTED
            marker = {"MAIN": "D", "N1": "s", "N2": "s", "N3": "s"}.get(model_id, "o")
            dot_interval(ax, y, estimate, lo, hi, color, marker)
            ax.text(col[0] + 0.015, y, f"{model_id}  {label}", va="center",
                    transform=transform, fontsize=8.5, color=color)
            ax.text(col[1], y, f"{count(row):,} / {count(row, 'events'):,}",
                    va="center", ha="right", transform=transform, fontsize=8.1)
            ax.text(col[2], y, f"{estimate:.3f} ({lo:.3f}–{hi:.3f})", va="center",
                    transform=transform, fontsize=8.5, color=color)
            release.track(row, panel="A", status="fall report RR", unit="ratio",
                          source_column="estimate|ci_low|ci_high|n|events|clusters|adjustment|chronic_source")
            y += 1.05
        y += 0.20
    ax.set_xscale("log")
    limits = (min(0.30, min(lowers) / 1.12), max(1.40, max(uppers) * 1.12))
    ax.set(xlim=limits, ylim=(y - 0.58, -1.05), yticks=[])
    ticks = [x for x in (0.2, 0.3, 0.5, 0.7, 1, 1.4, 2, 3) if limits[0] <= x <= limits[1]]
    ax.set_xticks(ticks, labels=[f"{value:g}" for value in ticks])
    ax.minorticks_off()
    ax.axvline(1, color="#737C83", lw=0.9, ls=(0, (3, 3)), zorder=1)
    ax.grid(axis="x", color=GRID, linewidth=0.65)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("RR for G4 versus G1 (log scale)")
    release.save(
        fig, "Figure 4. Primary contrast across fixed models",
        "Relative risks (RRs) for G4 versus G1 were estimated with modified Poisson models "
        "and 95% confidence intervals based on 2011 community-clustered variance. The N / reports "
        "column gives the participant count and the count reporting at least one fall. The base model "
        "uses demographic covariates, prior falls and baseline restless sleep frequency. "
        "Full health adjustment adds mood symptoms, chronic conditions, 100 m walking difficulty, "
        "distance vision and memory. F is the extended complete-case sample. R* relaxes walking "
        "completeness while retaining an available original chronic condition count and the "
        "other F criteria. N1 drops the walking term from F_RAW in F, N2 applies the same formula "
        "in R*, and N3 uses the base formula in R*. N1–N3 are post hoc. The samples overlap. "
        "G1 reported pain and restless sleep on at least 3 days in the previous week. G4 reported "
        "a negative response to the pain item and restless sleep on fewer than 3 days.",
        "Forest plot of nine estimates and intervals for G4 versus G1. Rows are grouped into "
        "primary and prior sensitivities, complete-case bridging and post hoc sensitivities.",
        placement="Main text, primary and sensitivity models",
        question="How does the G4 versus G1 RR vary across the fixed model and sample comparisons?",
        sample="P, F and R*; N and events shown per row",
        source="results_master.csv", identity="Expanded baseline figure", legacy="Old Figure 2 and Table 3",
    )


def figure_s1(release, missing):
    release.current_id = "FIGURE_S1"
    field_names = (
        ("walk100", "100 m walking difficulty"),
        ("mood9", "Nine-item mood score"),
        ("chronic12", "Harmonized condition count"),
        ("vision", "Distance vision"),
        ("memory", "Self-rated memory"),
        ("only_walk100", "Only walking difficulty missing"),
    )
    rows = [(one(missing, field=field_name), label) for field_name, label in field_names]
    denominator = count(rows[0][0], "denominator")
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.2))
    fig.subplots_adjust(left=0.39, right=0.92, top=0.96, bottom=0.18)
    upper = 0
    for y, (row, label) in enumerate(rows):
        if count(row, "denominator") != denominator:
            raise ValueError("Missingness figures require a common denominator.")
        n = count(row, "n_missing")
        if n > denominator:
            raise ValueError(f"Missing count exceeds its denominator: {row}")
        pct = 100 * n / denominator
        subset = row["field"] == "only_walk100"
        bar, = ax.barh(y, pct, height=0.55, color="#E3EDF1" if subset else BLUE,
                        edgecolor=BLUE, hatch="////" if subset else None)
        assert math.isclose(bar.get_width(), pct)
        ax.text(pct + 0.35, y, f"{n:,} ({pct:.1f}%)", ha="left", va="center",
                color=INK, fontsize=8.3)
        upper = max(upper, pct)
        release.track(
            row, panel="A", sample_id="P", status=row["field"], unit="percent",
            n_key="n_missing", source_column="n_missing|denominator", estimate=pct,
            role="post hoc missingness description",
        )
    ax.set(yticks=range(len(rows)), yticklabels=[label for _, label in rows],
           xlim=(0, max(5, math.ceil(upper / 5) * 5 + 3)), ylim=(len(rows) - 0.45, -0.55))
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_xlabel("Percentage of primary sample")
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    release.save(
        fig, "Supplementary Figure S1. Extended covariate missingness",
        f"The denominator for each bar is the primary sample (n = {denominator:,}). Participants "
        "can appear in more than one of the first five "
        "bars. The hatched bar is the subset with only 100 m walking difficulty missing among "
        "the extended covariates.",
        "Horizontal bars show missing counts and percentages for five extended covariates in P, "
        "with a hatched bar for the subset missing only walking difficulty.",
        placement="Additional file 1, missing data",
        question="Which extended fields limit completeness, and how many participants lack walking alone?",
        sample=f"P, n={denominator}",
        source="missing.csv", identity="New post hoc diagnostic", legacy="Visual companion to old Table S7",
    )


FIELD_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "prior_fall": "Prior fall",
    "sleep_severity_2011": "Baseline sleep frequency",
    "mood9": "Mood score",
    "chronic12": "Harmonized condition count",
    "walk100": "100 m walk",
    "vision": "Distance vision",
    "memory": "Memory",
}
COMPARISONS = (
    ("CLASS_vs_UNCLASS", "2015 classifiable vs unclassifiable", "o", BLUE),
    ("P_vs_NO_VALID_Y", "P vs no valid 2018 outcome", "s", ORANGE),
    ("F_vs_P_NOT_F", "F vs P outside F", "^", PURPLE),
    ("R0_vs_P_NOT_R0", "R0 vs P outside R0", "D", INK),
)


def figure_s2(release, selection):
    release.current_id = "FIGURE_S2"
    for comparison, *_ in COMPARISONS:
        if not select(selection, comparison=comparison):
            raise ValueError(f"Selection comparison absent: {comparison}")
    fields = {}
    for field_name, field_label in FIELD_LABELS.items():
        levels = list(dict.fromkeys(
            str(row["level"]) for row in select(selection, field=field_name)
        ))
        if not levels:
            raise ValueError(f"Selection field absent: {field_name}")
        binary_labels = {
            "sex": (2, "Female"),
            "prior_fall": (1, "Prior fall"),
            "walk100": (1, "Walking difficulty"),
            "sleep_severity_2011": (1, "Restless sleep 5–7 days"),
        }
        if field_name in binary_labels:
            display_level, label = binary_labels[field_name]
            (level,) = [level for level in levels if float(level) == display_level]
            fields[field_name, level] = label
            continue
        for level in levels:
            fields[field_name, level] = (
                field_label if not level else f"{field_label}, level {int(float(level))}"
            )
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 6.30))
    fig.subplots_adjust(left=0.375, right=0.96, top=0.98, bottom=0.18)
    limit = 0.0
    gaps = False
    offsets = (-0.31, -0.11, 0.11, 0.31)
    for y, ((field_name, level), label) in enumerate(fields.items()):
        for offset, (comparison, _, marker, color) in zip(offsets, COMPARISONS):
            found = select(selection, comparison=comparison, field=field_name, level=level)
            if not found:
                gaps = True
                continue
            row = one(found)
            smd = optional_num(row, "smd")
            if smd is None:
                gaps = True
            else:
                limit = max(limit, abs(smd))
                marker_artist, = ax.plot(
                    smd, y + offset, marker=marker, color=color, ms=4.3, linestyle="none"
                )
                assert math.isclose(marker_artist.get_xdata()[0], smd)
            release.track(
                row, panel="A", sample_id=comparison, status=field_name, unit="SMD",
                estimate_key="smd", source_column="smd|retained_n|excluded_n",
                n=str(count(row, "retained_n") + count(row, "excluded_n")),
                role="post hoc selection description" if smd is not None else "not computable",
            )
    limit = max(0.2, math.ceil((limit + 0.025) * 10) / 10)
    ax.set(xlim=(-limit, limit), yticks=range(len(fields)), yticklabels=list(fields.values()),
           ylim=(len(fields) - 0.43, -0.57))
    ax.xaxis.set_major_locator(MultipleLocator(0.2 if limit >= 0.5 else 0.1))
    ax.axvline(0, color="#737C83", lw=0.9, ls=(0, (3, 3)))
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_xlabel("Signed standardized mean difference")
    handles = [plt.Line2D([], [], marker=marker, color=color, lw=0, label=label, ms=5)
               for _, label, marker, color in COMPARISONS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.032),
               ncol=2, frameon=False, fontsize=8.0)
    release.save(
        fig, "Supplementary Figure S2. Baseline selection differences",
        "Signed standardized differences compare participants retained versus excluded at each "
        "selection step, using the available observations for each field. Marker shapes distinguish "
        "comparisons. Values to the right indicate a higher mean or proportion among retained "
        "participants. Each binary field is shown for one category. Vision and memory use "
        "ordered levels 1 through 5, from best to worst. The differences describe selection." + (
            " A missing marker indicates an unavailable or non-computable comparison." if gaps else ""
        ),
        "Four differently shaped markers compare baseline characteristics across four selection "
        "steps on a common signed scale centered at zero.",
        placement="Additional file 1, selection diagnostics",
        question="Which baseline characteristics differ between retained and excluded participants?",
        sample="Stage-specific retained and excluded groups; field-specific observed denominators",
        source="selection.csv", identity="New post hoc diagnostic", legacy="Extends old Table S3",
    )


def figure_s3(release, chronic):
    release.current_id = "FIGURE_S3"
    keys = ("SAME_ALL", "STATUS_DIFF_SAME_TOTAL", "TOTAL_DIFF", "DISPUTE_FLAG")
    rows = {key: one(chronic, status=key) for key in keys}
    counts = {key: count(row) for key, row in rows.items()}
    total = sum(counts[key] for key in keys[:3])
    if total < 1 or counts["DISPUTE_FLAG"] > total:
        raise ValueError("Invalid chronic condition source counts.")
    for key, row in rows.items():
        release.track(
            row, panel="A", sample_id="F", status=key, unit="percent", denominator=total,
            source_column="n", estimate=100 * counts[key] / total,
            role="post hoc source comparison",
        )
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.83))
    fig.subplots_adjust(left=0.415, right=0.95, top=0.97, bottom=0.23)
    labels = ("All 12 statuses agree", "Status differs, same total",
              "Total count differs", "At least one source dispute flag")
    positions = [0, 1, 2, 3.55]
    for i, (y, key) in enumerate(zip(positions, keys)):
        pct = 100 * counts[key] / total
        bar, = ax.barh(y, pct, color=BLUE if i < 3 else BLUE_LIGHT, edgecolor=BLUE,
                        hatch=None if i < 3 else "////", height=0.56)
        assert math.isclose(bar.get_width(), pct)
        if pct >= 76:
            ax.text(pct - 1.9, y, f"{counts[key]:,} ({pct:.1f}%)", va="center",
                    ha="right", color="white" if i < 3 else INK, fontsize=8.4)
        else:
            ax.text(pct + 1.6, y, f"{counts[key]:,} ({pct:.1f}%)", va="center", fontsize=8.4)
    ax.axhline(2.75, color=GRID, lw=0.9)
    ax.set(yticks=positions, yticklabels=labels, xlim=(0, 100), ylim=(4.18, -0.61))
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_major_formatter(PercentFormatter(100))
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Percentage of extended complete-case sample")
    release.save(
        fig, "Supplementary Figure S3. Chronic condition sources",
        "The top three bars are mutually exclusive and compare the original 2011 and Harmonized "
        f"definitions for 12 chronic conditions within the extended complete-case sample (F, n = {total:,}). "
        "The fourth bar is a separate source "
        "dispute indicator and may overlap any agreement category. The figure describes "
        "agreement between sources. It does not establish the accuracy of either source.",
        "Three bars describe agreement or disagreement between chronic condition sources. "
        "A separate hatched bar shows the proportion with at least one source dispute flag.",
        placement="Additional file 1, chronic source diagnostics",
        question="How often do the original and Harmonized condition statuses and totals differ in F?",
        sample=f"F, n={total}; common denominator",
        source="chronic.csv", identity="Refreshed source diagnostic", legacy="Visual companion to old Tables S8–S9",
    )


def render(results_dir, figures_dir):
    figures_dir.mkdir(parents=True, exist_ok=True)
    release = Release(figures_dir)
    results = read_table(results_dir, "results_master.csv")
    with mpl.rc_context(STYLE):
        figure1(release, read_table(results_dir, "flow.csv"))
        figure2(release, read_table(results_dir, "followup.csv"))
        figure3(release, results)
        figure4(release, results)
        figure_s1(release, read_table(results_dir, "missing.csv"))
        figure_s2(release, read_table(results_dir, "selection.csv"))
        figure_s3(release, read_table(results_dir, "chronic.csv"))
    release.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("figures_dir", type=Path)
    args = parser.parse_args()
    render(args.results_dir, args.figures_dir)


if __name__ == "__main__":
    main()
