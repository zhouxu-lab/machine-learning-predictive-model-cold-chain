# Run the models and numerical analyses

Run commands from the repository root using Python 3.12 and the dependencies in `code/requirements.txt`. The reference training environment is Windows; the controllers use Windows process APIs. Supply the private inputs described in [DATA_ACCESS.md](../DATA_ACCESS.md). A new public checkout contains no measurements, frozen splits, fitted weights, or individual predictions.

## Check inputs and model execution

```text
python -B verify_release.py
python -B code/verify_inputs.py
python -B code/verify_models.py
```

The first command checks public software integrity. The second checks frozen input and split hashes. The third checks source separation and neural forward execution. Only the first can run without the private inputs. These checks do not establish an independent refit.

For saved benchmark prediction rows in a separate local directory:

```text
python -B code/verify_results.py --data PATH_TO_SAVED_DYNAMIC_PREDICTIONS
```

This read-only check needs NumPy and pandas; `code/requirements-checks.txt` lists those lighter dependencies. It compares the saved individual predictions with the reported metrics.

## Fit the seven model conditions

Use a separate working copy with the required inputs and no archived selection/completion markers in `results/Training`:

```text
python -B code/train_all.py --full
```

The wrapper runs pilots and numerical preflight checks, source-grouped inner model selection, and final fitting with three seeds per outer fold. It then aggregates held-out predictions, including starting-count persistence as the unfitted reference. Omitting `--full` runs pilots/preflight only. Full fitting is a substantial CPU/GPU job; run it deliberately.

New model outputs are written under `results/Training`; benchmark summaries are written under `results/Predictions/Dynamic`. Controller and pilot bookkeeping may also appear next to the source and is ignored. Existing completed training state is rejected by `--full`. Do not erase a retained archive to bypass that guard. Controller resume is appropriate only for a compatible interrupted run, including matching input contracts and source-code identity. Resume an interrupted fit with the same software release that started it; folder-path updates change the recorded source hashes.

To reaggregate a completed fresh fit:

```text
python -B code/Modeling/InitialConcentration/aggregate_clean_models.py --from-training
```

Without `--from-training`, that script reads separately supplied archived seed predictions under `results/Predictions/Fitted`, not the newly trained outputs. The catalog and input/split contracts must match the chosen predictions. A current ComBase download alone does not recreate the reviewed source mapping and frozen inputs.

## Initial-concentration sensitivity

Supply the additional normalized preparation tables and fitted outputs described in DATA_ACCESS before running:

```text
python -B code/Modeling/InitialConcentration/build_initial_count_sensitivity_cohort.py
python -B code/Modeling/InitialConcentration/run_initial_sensitivity.py
python -B code/Modeling/NeuralODE/score_initial_node.py
python -B code/Modeling/InitialConcentration/run_measured_anchor.py
python -B code/Modeling/InitialConcentration/score_measured_anchor_node.py
python -B code/Modeling/InitialConcentration/summarize_initial_sensitivity.py
python -B code/Modeling/InitialConcentration/summarize_measured_anchor.py
```

The hypothetical initial counts are assumptions, not observed counts or verified detection limits. These scorers use fitted model outputs and can skip existing fold predictions. Use a separate output working copy when changing fitted models; preserve the reference results.

Additional descriptive summaries are available through `organism_results.py` and `paired_model_intervals.py` in `Modeling/InitialConcentration`. They require the matching saved predictions; they do not perform new fitting.

## Constant-temperature input reliance

This is a separate analysis cohort:

```text
python -B code/Modeling/InputReliance/fit_static_models.py
python -B code/Modeling/InputReliance/analyze.py
python -B code/Modeling/InputReliance/ablate.py
python -B code/Modeling/InputReliance/summarize_ablations.py
python -B code/Modeling/InputReliance/summarize_reliance.py
```

`save_inputs.py` writes the explicit constant-temperature input table from the prepared features. Input reliance is predictive and does not establish causal effects.

## Input preparation

`Modeling/Preprocessing` retains parsers, curation, feature construction, and the final clean-training exclusion. These scripts require the original normalized tables and source metadata; they do not fetch or recreate the reviewed archive. Curation refuses to overwrite an existing manifest. Rebuild only in a disposable working copy, verify resulting input hashes, and investigate mismatches rather than bypassing them. The manuscript's supplementary tables are supplied separately and are not training inputs.
