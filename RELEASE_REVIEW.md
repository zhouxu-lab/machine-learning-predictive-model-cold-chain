# Release scope and verification

This is a submission and review software release prepared on 23 October 2026.
It includes the analysis code and exactly three accompanying supplementary CSV
tables: model settings, aggregate organism performance, and selected model/fold
configurations. It does not include the manuscript, figures, individual source
observations, record-level processed data, source correspondence, frozen split
identifiers, fitted model weights, prediction rows, or third-party Predictor
source. The code's MIT license does not relicense those third-party materials or
the separately shared research tables.

`Code_Manifest.csv` records release paths, sizes, and SHA-256 values. Run
`python -B verify_release.py` to check manifest integrity, Python syntax,
credential/private-path patterns, and the supplementary table allowlist.
Keep privately staged files ignored and outside any public commit or release ZIP.

The private preparation checks confirmed frozen input/split identities, shared
evaluation rows, summary metrics, source separation, and real-record neural
forward passes. Representative saved model artifacts were reloaded successfully.
Targeted training-state/path repairs were checked using real-input tree pilots,
a fail-fast archived-state check, and a hybrid dependency check stopped before
fitting. Model equations, fit settings, and archived scientific results were
preserved. These checks do not certify complete independent retraining.

For a fresh fit, use a separate checkout and approved private `--scope training`
staging. `train_all.py --full` refuses archived selection/completion markers
before starting pilots. Full numerical reproduction requires the frozen private
inputs described in `DATA_ACCESS.md`; public code and summary tables alone are
insufficient. A new live ComBase download may differ.

The software citation identifies this review release. No article acceptance,
article DOI, software DOI, or public access to excluded private data is asserted.
