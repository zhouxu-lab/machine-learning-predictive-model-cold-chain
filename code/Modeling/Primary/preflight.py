from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'data').is_dir())
sys.path.insert(0,str(_PACKAGE/'code'))
sys.path.insert(0,str(_PACKAGE/'code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""Fail-fast checks for the isolated censor-safe all-model refit."""
from pathlib import Path
import json
import pickle
import sys
import numpy as np
import pandas as pd
from common import ROOT, BASE, contract, identity, splits, sha

def main():
    ident = identity()
    cfg = contract()
    assert list(cfg["configs"]) == ["backbone", "gru_abs", "transformer_abs", "mul", "rf", "gbm"]
    assert cfg["splits_sha256"] == sha((DATA/"Splits_and_Configurations"/'Primary'/'splits.json'))
    features = pd.read_parquet(BASE / "features.parquet")
    with (BASE / "deep_prepared.pkl").open("rb") as f:
        recs, _, _ = pickle.load(f)
    ids = {int(r["record_id"]) for r in recs}
    assert ids == set(map(int, features.record_id.unique()))
    assert not np.isclose(features.logc, -.01, rtol=0, atol=1e-8).any()
    assert not np.isclose(features.logc0_obs, -.01, rtol=0, atol=1e-8).any()
    assert all(not np.isclose(np.asarray(r["obs_y"]) + r["y0"], -.01, rtol=0, atol=1e-5).any() for r in recs)
    sources = {int(r["record_id"]): str(r["source_key"]) for r in recs}
    for fold in splits().values():
        assert set(map(sources.get, fold["train"])).isdisjoint(set(map(sources.get, fold["test"])))
        for inner in fold["inner"].values():
            assert set(map(sources.get, inner["train"])).isdisjoint(set(map(sources.get, inner["valid"])))
    pilots = ["rf", "gbm", "gru_abs", "transformer_abs", "backbone", "mul"]
    assert all((ROOT / "pilot" / name / "complete.json").exists() for name in pilots)
    report = {"status": "passed", "contract_sha256": ident, "feature_rows": len(features),
              "records": len(recs), "source_folds": len(splits()), "pilots": pilots,
              "python": sys.version.split()[0]}
    path = ROOT / "qa" / "checks.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
