"""Copy approved private inputs into ignored local folders; never download/upload.

Use --dry-run to list paths and bytes. Existing different files are never replaced.
Do not include the resulting ignored folders when publishing the code candidate.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil


ROOT = Path(__file__).resolve().parent
SCOPES = {
    "checks": [
        "06_Results/Predictions/Dynamic/censor_safe_matched_points.csv",
        "06_Results/Predictions/Dynamic/censor_safe_model_metrics.csv",
    ],
    "training": [
        "05_Data/Processed/Training",
        "05_Data/Processed/record_catalog.csv",
        "05_Data/Splits_and_Configurations/Primary",
        "05_Data/Splits_and_Configurations/NeuralODE",
    ],
    "figures": [
        "05_Data/Processed/Archive/profiles.parquet",
        "06_Results/Figure_Source_Data",
        "06_Results/Predictions/Dynamic",
        "06_Results/Predictions/InputReliance",
    ],
    "full": ["05_Data", "06_Results"],
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plan(source, scope):
    source = Path(source).resolve()
    if source == ROOT or not (source / "04_Code").is_dir():
        raise ValueError("--source-root must be a separate private package containing 04_Code")
    rows = []
    for relative in SCOPES[scope]:
        selected = source / relative
        if not selected.exists():
            raise FileNotFoundError(f"Required private input missing: {relative}")
        files = sorted(selected.rglob("*")) if selected.is_dir() else [selected]
        for path in files:
            if not path.is_file():
                continue
            rel = path.relative_to(source)
            target = ROOT / rel
            if not target.resolve().is_relative_to(ROOT):
                raise ValueError("Private staging destination escapes candidate")
            # Preserve the candidate's explanatory placeholder files.
            if rel.as_posix() in ("05_Data/README.md", "06_Results/README.md"):
                continue
            rows.append((path, target, rel.as_posix()))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--scope", choices=SCOPES, default="checks")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows = plan(args.source_root, args.scope)
    summary = {"scope": args.scope, "files": len(rows),
               "bytes": sum(src.stat().st_size for src, _, _ in rows),
               "dry_run": args.dry_run}
    if args.dry_run:
        summary["paths"] = [rel for _, _, rel in rows]
    else:
        hashes = []
        # Inspect every existing destination before beginning a copying pass.
        for src, target, rel in rows:
            digest = sha256(src)
            if target.exists() and sha256(target) != digest:
                raise FileExistsError(f"Different local file exists; use a new working copy: {rel}")
            hashes.append(digest)
        for (src, target, rel), digest in zip(rows, hashes):
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
            if sha256(target) != digest:
                raise IOError(f"Copied file hash mismatch: {rel}")
        summary["status"] = "PASS"
        summary["copied_file_hashes"] = {rel: digest for (_, _, rel), digest in zip(rows, hashes)}
        (ROOT / "local_staging.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
