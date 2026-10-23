"""Read-only checks of the frozen training inputs and split contracts.

This uses only the Python standard library. It does not deserialize pickle
files, train models, or change data. Source separation and neural forward
execution are checked separately by verify_models.py.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root):
    root = Path(root).resolve()
    data = root / "data"
    checks = []
    split_hashes = []
    for family in ("Primary", "NeuralODE"):
        folder = data / "Splits_and_Configurations" / family
        contract = json.loads((folder / "contract.json").read_text(encoding="utf-8"))
        for name, expected in contract["input_hashes"].items():
            path = data / "Processed" / "Training" / name
            actual = sha256(path)
            if actual != expected:
                raise ValueError(f"{family} input hash mismatch: {name}")
            checks.append({"family": family, "file": path.relative_to(root).as_posix(),
                           "sha256": actual})
        path = folder / "splits.json"
        actual = sha256(path)
        if actual != contract["splits_sha256"]:
            raise ValueError(f"{family} split hash mismatch")
        split_hashes.append(actual)
        checks.append({"family": family, "file": path.relative_to(root).as_posix(),
                       "sha256": actual})
    if len(set(split_hashes)) != 1:
        raise ValueError("Primary and NeuralODE do not use identical frozen splits")
    versions = {}
    for name in ("numpy", "pandas", "pyarrow", "scikit-learn", "lightgbm", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {"status": "PASS", "scope": "Input and split hashes only; no independent retraining",
            "python": platform.python_version(), "installed_packages": versions,
            "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--out", type=Path, help="Optional report; source files stay unchanged")
    args = parser.parse_args()
    result = verify(args.root)
    output = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
