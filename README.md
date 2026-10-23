# ComBase cold-chain modeling code and supplementary tables

Submission and review release prepared on 23 October 2026 for **Machine learning
and hybrid models for predicting microbial growth in cold food supply chains**,
by Xu Zhou. Repository: [zhouxu-lab/combase-coldchain-sft](https://github.com/zhouxu-lab/combase-coldchain-sft).

`04_Code` contains the current censor-safe model definitions, training controllers,
cohort and scoring scripts, and the 22 October figure builders. The software
license is MIT. [Supplementary_Data](Supplementary_Data/README.txt) provides the
three accompanying model-settings, aggregate organism-performance, and selected
configuration tables. Measurements, record-level tables, source correspondence, split
identifiers, fitted weights, manuscripts, figure assets, and third-party ComBase
Predictor files are excluded. The MIT code license does not relicense ComBase
observations or these separately shared research tables. See [DATA_ACCESS.md](DATA_ACCESS.md).

## Start with the smallest check

From this directory, using Python 3.12:

```text
python -B verify_release.py
python -m pip install -r 04_Code/requirements-checks.txt
python -B 04_Code/verify_results.py --data PATH_TO_PRIVATE_DYNAMIC_PREDICTIONS
```

The first command uses only the standard library and checks the candidate's
manifest, Python syntax, and file scope. The last command reads
`censor_safe_matched_points.csv` and `censor_safe_model_metrics.csv` from the supplied
directory without modifying them. Those prediction rows are not included in this
public release and require an approved private access route.

The expanded benchmark has 511 records, 4,247 later quantified observations, and
ten held out sources per approach. LightGBM source balanced MAE is 1.1143 log units;
pooled observation MAE is 1.2162. The comparison includes seven fitted conditions
and starting-count persistence. Code verification does not resolve manuscript
editorial decisions or establish independent retraining.

## Stage private inputs for additional checks

The placeholder `05_Data`, `06_Results`, and `03_Figures` directories preserve the
portable package layout. Keep all staged files untracked. The helper copies only
the requested local folders, checks copied file hashes, and refuses to replace a
different existing file. It does not download or upload anything.

```text
python -B stage_private_inputs.py --source-root PATH_TO_PRIVATE_REVIEW_PACKAGE --scope checks
python -B 04_Code/verify_results.py
python -B stage_private_inputs.py --source-root PATH_TO_PRIVATE_REVIEW_PACKAGE --scope training
python -B 04_Code/verify_inputs.py
python -B 04_Code/verify_models.py
```

`--scope figures` stages the final plotting inputs. `--scope full` stages all local
data and results, including about 4.9 GB of final model weights; use it only in a
private working copy. Use `--dry-run` to inspect any staging plan first. The helper
does not erase outputs from an earlier working run.

For a fresh refit, start from a separate checkout of this repository and
stage **only `--scope training`**, which includes the record catalog required for
the final summary. Then run `python -B 04_Code/train_all.py --full`. The command
refuses existing selected/completed training state before starting any pilot or
fit. Staging the full saved archive is the inference/review workflow, not a fresh
training destination. Never erase the delivered archive to bypass this check.

## Figures, saved results, and training

Install `04_Code/requirements.txt` in an environment outside this repository. The
reference environment was Windows, Python 3.12.13, and PyTorch 2.11.0 with CUDA 12.8.
For that GPU stack, install torch from its CUDA 12.8 wheel index before installing
the other pinned requirements. The training schedulers use Windows process APIs.

After private staging, `python -B 04_Code/Figures/generate_all.py` rebuilds seven
600 dpi TIFFs and associated source tables. It writes local files and does not
insert them into Word. `04_Code/REPRODUCE.md` describes the individual analysis
commands and full training sequence. Run writing commands in a working copy.

The local review archive includes 105 final weights and the saved prediction
rows. A complete independent training run from a fresh checkout has not been
certified. The exact manuscript results are defined by the frozen snapshot,
reviewed source grouping, split contracts, and saved predictions. The raw parsers
alone do not rebuild the reviewed source mapping and normalized intermediate
tables. A new live download may differ from the retained snapshot.

This candidate has three targeted training-control/portability repairs relative
to the inherited October code: pilots use their own state directory before any
completion check; hybrid fits load their backbone from the canonical run folder;
and the full-run wrapper rejects archived training state before doing work and
creates the final summary directory. Model equations, fit settings, data, saved
weights, and predictions are preserved. The private review archive records the
before copies, exact diff, and targeted verification scope.

## Supplementary tables and release scope

The public supplementary tables are `model_settings.csv` (seven model conditions),
`organism_performance_full.csv` (17 aggregate organism rows), and
`selected_configurations.csv` (35 model/fold selections). They contain settings
and aggregate results, with no individual observations or record/source keys.
The code manifest includes their sizes and SHA-256 hashes.

[RELEASE_REVIEW.md](RELEASE_REVIEW.md) records the release scope and verification
limits. The citation file identifies the software release. Private inputs and
the exact supporting prediction rows require separate access arrangements;
public code and summary tables alone do not provide complete numerical
reproduction.
