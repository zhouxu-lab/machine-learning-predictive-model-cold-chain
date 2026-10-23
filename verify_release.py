"""Read-only validation of code and approved supplementary summary tables.

This checks the frozen code manifest, Python syntax, and file scope. It does
not validate model results or certify a clean independent training run.
"""
from pathlib import Path
import ast
import csv
import hashlib
import json
import re


ROOT = Path(__file__).resolve().parent
PRIVATE = {"05_Data", "06_Results"}
EXCLUDED_NAMES = {"Code_Manifest.csv", "local_staging.json"}
REMOVED_PUBLIC_FOLDERS = {"03_figures", "figures", "figure_source_data", "officialpredictor"}
REMOVED_PUBLIC_FILES = {
    "archive_labels.py", "audit_panel_alignment.py", "figure_export.py",
    "build_archive_counts.py", "run_official_predictor.py",
    "run_official_predictor.cjs", "stage_private_inputs.py",
}
PUBLIC_TABLES = {
    "Supplementary_Data/model_settings.csv": (7, ["Model", "Fixed settings", "Four candidate settings"]),
    "Supplementary_Data/organism_performance_full.csv": (17, ["Organism label", "Records / sources", "Later counts", "MAE", "Underpred. >0.5", "Missed / growth"]),
    "Supplementary_Data/selected_configurations.csv": (35, ["model", "outer_fold", "configuration_index", "final_epochs", "configuration_json"]),
}


def release_files():
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(part in {".git", ".venv", "venv", "__pycache__"} for part in rel.parts):
            continue
        if rel.as_posix() in EXCLUDED_NAMES:
            continue
        if rel.parts[0] in PRIVATE and rel.as_posix() != f"{rel.parts[0]}/README.md":
            continue
        if (any(part.casefold() in REMOVED_PUBLIC_FOLDERS for part in rel.parts[:-1])
                or rel.name.casefold() in REMOVED_PUBLIC_FILES):
            raise ValueError(f"Figure, external Predictor, or staging code outside public scope: {rel}")
        yield path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    with (ROOT / "Code_Manifest.csv").open(encoding="utf-8", newline="") as stream:
        expected = list(csv.DictReader(stream))
    actual = {path.relative_to(ROOT).as_posix(): path for path in release_files()}
    names = [row["file"] for row in expected]
    if len(set(names)) != len(names) or set(names) != set(actual):
        raise ValueError("Code manifest does not match the current release file set")
    if not set(PUBLIC_TABLES).issubset(actual):
        raise ValueError("The approved supplementary tables are incomplete")
    py_count = 0
    forbidden = {".parquet", ".pkl", ".pickle", ".pt", ".pth", ".joblib",
                 ".docx", ".pdf", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".svg",
                 ".gif", ".webp", ".bmp", ".avif", ".eps", ".ps", ".psd", ".ai",
                 ".zip", ".xml", ".html", ".jsonl"}
    credentials = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{30,})")
    private_path = re.compile(r"[A-Za-z]:[\\/]Users[\\/]", re.IGNORECASE)
    for row in expected:
        rel = row["file"]
        path = actual[rel]
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            raise ValueError(f"Code file differs from manifest: {rel}")
        if path.suffix.lower() in forbidden:
            raise ValueError(f"Private or unsupported asset in public release: {rel}")
        if path.suffix.lower() == ".csv":
            if rel not in PUBLIC_TABLES:
                raise ValueError(f"CSV outside approved supplementary scope: {rel}")
            expected_rows, expected_header = PUBLIC_TABLES[rel]
            with path.open(encoding="utf-8-sig", newline="") as stream:
                table = list(csv.reader(stream))
            if table[0] != expected_header or len(table) != expected_rows + 1:
                raise ValueError(f"Supplementary schema or row count differs: {rel}")
            if any(len(row) != len(expected_header) for row in table):
                raise ValueError(f"Supplementary row width differs: {rel}")
        if path.suffix.lower() in {".py", ".md", ".txt", ".cff", ".cjs", ".csv"}:
            source = path.read_text(encoding="utf-8-sig")
            if credentials.search(source) or private_path.search(source):
                raise ValueError(f"Credential-like token or personal absolute path detected: {rel}")
            if path.suffix == ".py":
                ast.parse(source, filename=rel)
                py_count += 1
    print(json.dumps({"status": "PASS", "manifest_files": len(expected),
                      "python_files_syntax_checked": py_count,
                      "approved_supplementary_tables": len(PUBLIC_TABLES),
                      "scope": "Code/summary-table integrity, syntax and scope; no independent retraining",
                      "private_staging": "Excluded from code manifest; keep ignored and do not publish"}, indent=2))


if __name__ == "__main__":
    main()
