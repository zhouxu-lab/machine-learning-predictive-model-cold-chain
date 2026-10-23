from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'data').is_dir())
sys.path.insert(0,str(_PACKAGE/'code'))
sys.path.insert(0,str(_PACKAGE/'code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""Verify the current censor-safe benchmark from saved prediction rows.

This does not train models or overwrite research data. Use --data to point to
the folder containing censor_safe_matched_points.csv and its metrics CSV.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


def verify(folder):
    points = pd.read_csv(folder / 'censor_safe_matched_points.csv')
    expected = pd.read_csv(folder / 'censor_safe_model_metrics.csv')
    models = {'rf', 'gbm', 'gru_abs', 'transformer_abs', 'node', 'mul', 'backbone', 'persistence'}
    assert set(points.model) == models, 'Incomplete model comparison'
    assert not points.duplicated(['record_id', 'time_h', 'model']).any()
    assert np.isfinite(points[['observed_log', 'origin_log', 'predicted_log']]).all().all()
    keys = ['record_id', 'time_h']
    reference = points[points.model.eq('gbm')].sort_values(keys).reset_index(drop=True)
    for model, group in points.groupby('model'):
        group = group.sort_values(keys).reset_index(drop=True)
        assert len(group) == 4247 and group.record_id.nunique() == 511
        assert group.source_key.nunique() == 10
        pd.testing.assert_frame_equal(group[keys + ['source_key', 'strict_refrigerated']],
                                      reference[keys + ['source_key', 'strict_refrigerated']])
        np.testing.assert_allclose(group[['observed_log', 'origin_log']],
                                   reference[['observed_log', 'origin_log']], atol=1e-6, rtol=0)
    error = points.predicted_log - points.observed_log
    np.testing.assert_allclose(error, points.error_log, atol=1e-12, rtol=0)
    np.testing.assert_allclose(error.abs(), points.absolute_error, atol=1e-12, rtol=0)
    rows = []
    for cohort, data in [('expanded', points), ('strict_refrigerated', points[points.strict_refrigerated])]:
        for model, g in data.groupby('model'):
            e = g.predicted_log - g.observed_log
            by_source = g.assign(a=e.abs(), e2=e**2).groupby('source_key')[['a', 'e2']].mean()
            row = dict(cohort=cohort, model=model, later_points=len(g), records=g.record_id.nunique(),
                       sources=g.source_key.nunique(), mae=e.abs().mean(),
                       source_balanced_mae=by_source.a.mean(), rmse=np.sqrt((e**2).mean()),
                       source_balanced_rmse=np.sqrt(by_source.e2.mean()), bias_log=e.mean(),
                       underprediction_over_0p5=(e < -.5).mean())
            target = expected[expected.cohort.eq(cohort) & expected.model.eq(model)]
            assert len(target) == 1
            for metric in (set(row) & set(expected.columns)) - {'model', 'cohort'}:
                assert np.isclose(row[metric], target.iloc[0][metric], atol=1e-9, rtol=0), (cohort, model, metric)
            rows.append(row)
    persistence = points[points.model.eq('persistence')]
    np.testing.assert_array_equal(persistence.predicted_log, persistence.origin_log)
    return {'status': 'PASS', 'scope': 'Saved prediction and metric verification; no independent retraining',
            'models': len(models), 'matched_points_per_model': 4247, 'metrics': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=RESULTS/'Predictions/Dynamic')
    parser.add_argument('--out', type=Path, help='Optional JSON verification report; input files remain unchanged')
    args = parser.parse_args()
    result = verify(args.data)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
