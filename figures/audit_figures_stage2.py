"""Check the publication figures against frozen aggregates and their rendered annotations."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import subprocess

from PIL import Image
from pypdf import PdfReader


EXPECTED = {
    "FIGURE_1": 9, "FIGURE_2": 16, "FIGURE_3": 5, "FIGURE_4": 9,
    "FIGURE_S1": 6, "FIGURE_S2": 68, "FIGURE_S3": 4,
}
NUMERIC = {"n", "events", "clusters", "denominator", "estimate", "ci_low", "ci_high"}
KEYS = {
    "flow.csv": ("stage",),
    "followup.csv": ("group_id", "status"),
    "results_master.csv": ("model_id", "estimand", "group_id", "contrast_id"),
    "missing.csv": ("field",),
    "selection.csv": ("comparison", "field", "level"),
    "chronic.csv": ("status",),
}
DISPLAY_RULE = {
    "FIGURE_1": "Frozen counts as whole people",
    "FIGURE_2": "Bar width = 100 × n / group denominator, annotation = whole count",
    "FIGURE_3": "Proportion × 100 to percentage; difference × 100 to percentage points; print 1 decimal",
    "FIGURE_4": "Log-axis RR and limits at frozen precision, print 3 decimals; counts as whole people",
    "FIGURE_S1": "Bar width = 100 × n_missing / primary sample denominator; print percent to 1 decimal",
    "FIGURE_S2": "Marker x coordinate = frozen signed SMD; retain field-level denominators",
    "FIGURE_S3": "Bar width = 100 × n / complete-case denominator; print percent to 1 decimal",
}


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_sources(results, figures):
    data = read(figures / "FIGURE_SOURCE_LONG.csv")
    counts = Counter(row["figure_id"] for row in data)
    assert counts == EXPECTED, f"Unexpected provenance row counts {counts}"
    tables = {name: read(results / name) for name in {r["source_table"] for r in data}}
    for row in data:
        source = tables[row["source_table"]][int(row["source_row"]) - 2]
        fig_id = row["figure_id"]
        extra = {}
        if fig_id == "FIGURE_2":
            extra.update(estimate=source["percent"], events=source["n"] if source["status"] == "FALL" else "")
        elif fig_id in {"FIGURE_S1", "FIGURE_S3"}:
            n = source.get("n_missing", source.get("n"))
            extra.update(n=n, estimate=100 * int(n) / int(source["denominator"]))
        elif fig_id == "FIGURE_S2":
            extra.update(estimate=source["smd"], n=int(source["retained_n"]) + int(source["excluded_n"]))
        for key in NUMERIC:
            expected = extra.get(key, source.get(key, ""))
            actual = row[key]
            if expected == "":
                assert not actual, (fig_id, key, actual)
            else:
                assert math.isclose(float(expected), float(actual), abs_tol=1e-12, rel_tol=0), (fig_id, key, expected, actual)
        for key in ("model_id", "group_id", "standardization_reference"):
            if key in source:
                assert row[key] == source[key], (fig_id, key, source, row)
        key_names = KEYS[row["source_table"]]
        row["source_row_key"] = " | ".join(f"{key}={source[key]}" for key in key_names)
        row["display_rule"] = DISPLAY_RULE[fig_id]
    return dict(counts), data


def formatted_estimate(row):
    fields = ("estimate", "ci_low", "ci_high")
    point, lower, upper = [float(row[key]) for key in fields]
    if row["figure_id"] == "FIGURE_4":
        return f"{point:.3f} ({lower:.3f}–{upper:.3f})"
    point, lower, upper = [100 * value for value in (point, lower, upper)]
    if row["status_or_outcome"] == "G4 minus G1":
        return f"{point:+.1f} percentage points (95% CI {lower:+.1f} to {upper:+.1f})".replace("-", "−")
    return f"{point:.1f}% ({lower:.1f}%–{upper:.1f}%)"


def verify_annotations(pdf_path, rows, text):
    figure = rows[0]["figure_id"]
    layout = subprocess.check_output(["pdftotext", "-layout", pdf_path, "-"], text=True)
    if figure == "FIGURE_2":
        for group in dict.fromkeys(row["group_id"] for row in rows):
            group_rows = [row for row in rows if row["group_id"] == group]
            line = next(line for line in layout.splitlines() if re.match(rf"\s*{group}\b", line))
            denominator = group_rows[0]["denominator"]
            assert f"n = {denominator})" in line
            printed_counts = line.split(")", 1)[1].split()
            assert printed_counts == [row["n"] for row in group_rows]
            for row in group_rows:
                row["display_annotation"] = f"{group} {row['status_or_outcome']}: {row['n']} of {denominator}"
        return
    for row in rows:
        if figure == "FIGURE_1":
            annotation = f"n = {int(row['n']):,}"
        elif figure in {"FIGURE_S1", "FIGURE_S3"}:
            annotation = f"{int(row['n']):,} ({float(row['estimate']):.1f}%)"
        elif figure == "FIGURE_S2":
            row["display_annotation"] = f"x = {row['estimate']}; no printed SMD"
            continue
        else:
            annotation = formatted_estimate(row)
            if figure == "FIGURE_4":
                label = row["model_id"]
                line = next(line for line in layout.splitlines() if re.match(rf"\s*{label}\b", line))
                counts = f"{int(row['n']):,} / {int(row['events']):,}"
                assert counts in line and annotation in line, (row, line)
                annotation = f"{counts} | {annotation}"
                row["display_annotation"] = annotation
                continue
            if row["status_or_outcome"] == "fall report probability":
                section = re.search(rf"(?ms)^\s*{row['group_id']}\s.*?(?=^\s*G[1-4]\s|\Z)", layout)
                assert section and annotation in section.group(), (row, layout)
        assert annotation in text, (figure, annotation)
        row["display_annotation"] = annotation


def check_formats(figures, data):
    files = {}
    for fig_id in EXPECTED:
        pdf_path, png_path = (figures / f"{fig_id}.{ext}" for ext in ("pdf", "png"))
        reader = PdfReader(pdf_path, strict=True)
        assert len(reader.pages) == 1
        page = reader.pages[0]
        assert "/Font" in page["/Resources"]
        text = page.extract_text()
        assert text.strip(), fig_id
        rows = [r for r in data if r["figure_id"] == fig_id]
        verify_annotations(pdf_path, rows, text)
        pdf_points = [float(page.mediabox.width), float(page.mediabox.height)]
        pdf_mm = [dimension / 72 * 25.4 for dimension in pdf_points]
        assert abs(pdf_mm[0] - 170) < 0.01
        assert pdf_mm[1] < 225
        assert pdf_path.stat().st_size < 10_000_000
        font_report = subprocess.check_output(["pdffonts", pdf_path], text=True)
        font_rows = font_report.splitlines()[2:]
        assert font_rows and all(row.split()[-5] == "yes" for row in font_rows), font_report
        with Image.open(png_path) as img:
            assert img.format == "PNG"
            assert min(img.info["dpi"]) >= 300
            assert all(abs(pixels - float(points) / 72 * img.info["dpi"][i]) <= 2
                       for i, (pixels, points) in enumerate(zip(img.size, (page.mediabox.width, page.mediabox.height))))
            files[fig_id] = {
                "pixels": img.size, "dpi": img.info["dpi"],
                "pdf_points": pdf_points,
                "pdf_mm": pdf_mm,
                "embedded_fonts": [row.split()[0] for row in font_rows],
                "pdf_sha256": digest(pdf_path), "png_sha256": digest(png_path),
            }
            img.verify()
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("figures", type=Path)
    p.add_argument("output_json", type=Path)
    args = p.parse_args()
    counts, data = check_sources(args.results, args.figures)
    files = check_formats(args.figures, data)
    with (args.output_json.parent / "FIGURE_NUMERIC_MAP.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    receipt = {
        "status": "PASS", "python": platform.python_version(),
        "figure_source_row_counts": counts,
        "provenance_float_absolute_tolerance": 1e-12,
        "numeric_pdf_annotation_check": "PASS",
        "format_files": files,
        "frozen_source_sha256": {name: digest(args.results / name) for name in KEYS},
        "figure_source_map_sha256": digest(args.figures / "FIGURE_SOURCE_LONG.csv"),
    }
    args.output_json.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
