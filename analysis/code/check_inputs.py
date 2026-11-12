"""Check authorized input identities and output paths without fitting a model."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
CODEBOOK = {
    "file": "Harmonized CHARLS/Harmonized_CHARLS_D.pdf",
    "sha256": "40957f6e14b0f60aac7570aa59ed7a0aa27a1870cbc8b90b3b48207c74603a51",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--private", required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    private = args.private.resolve()
    if out.is_relative_to(private) or private.is_relative_to(out):
        parser.error("--out and --private must be separate and cannot be nested.")
    for path in (out, private):
        if path.is_relative_to(REPOSITORY) or path.exists() or not path.parent.is_dir():
            parser.error("Each destination must be a fresh directory outside the repository with an existing parent.")
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    raw = Path(config["data_root"]).expanduser()
    if not raw.is_absolute() or not raw.is_dir():
        parser.error("data_root must be an existing absolute directory.")
    pins = json.loads((HERE / "reference/input_lock.json").read_text(encoding="utf-8-sig"))
    for pin in [item for item in pins if item["used_in"] == "analysis"] + [CODEBOOK]:
        source = raw / pin["file"]
        if not source.is_file():
            parser.error(f"Missing authorized file: {pin['file']}")
        with source.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != pin["sha256"]:
            parser.error(f"Release hash differs: {pin['file']}")
    import stage1  # Imports the runtime and inherited modules without starting analysis.

    if len(stage1.model_spec()) != 9:
        parser.error("The frozen model specification did not load as expected.")
    print("Verified 11 data files and the Harmonized Version D codebook.")
    print("Analysis dependencies: " + ", ".join(
        f"{package} {version(package)}"
        for package in ("numpy", "pandas", "scipy", "patsy", "statsmodels", "openpyxl", "matplotlib")
    ))


if __name__ == "__main__":
    main()
