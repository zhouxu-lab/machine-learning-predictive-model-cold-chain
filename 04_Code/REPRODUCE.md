# Reproduce the saved analyses and review figures

All commands below run from this candidate's root, the folder containing `04_Code`, `05_Data`, and `06_Results`, after approved private staging described in `../README.md`. Python 3.12 is the reference runtime. Use the pinned `04_Code/requirements.txt`; the original training used Windows and PyTorch with CUDA 12.8. The training controllers use Windows process locking. The official USDA Predictor comparator additionally requires Node.js because its supplied implementation is JavaScript.

The copied plotting scripts include the separately revised 22 October Figures 2–5. The manuscript baseline and review reports are held in the separate private submission package. Figure files, captions, and the manuscript need to be assessed together; running code does not confirm that every editorial or submission issue is resolved. Scientific model implementations were preserved. Private inputs, stored predictions, weights, and acquisition provenance must be obtained separately.

## Environment

Create a Python environment outside this deliverable folder, then install the requirements. For the original GPU stack, install `torch==2.11.0` from `https://download.pytorch.org/whl/cu128` before installing the remaining requirements. Do not place an environment or package cache inside the submission package.

## Rebuild figures from final saved results

For a small read-only check, install `04_Code/requirements-checks.txt` and run the first two commands below. Input checks use only the Python standard library; saved-prediction checks need NumPy and pandas. Model forward checks and plotting require the full environment.

```
python -B 04_Code/verify_inputs.py
python -B 04_Code/verify_results.py
python -B 04_Code/verify_models.py
python -B 04_Code/Figures/generate_all.py
```

Individual entry scripts are `figure_1.py` through `figure_5.py`, `figure_s1.py`, and `figure_s2.py`. Each writes one 600 dpi TIFF in `03_Figures`, and some also refresh plotted source tables and alignment reports in `06_Results`. Common functions are in `04_Code/Shared`. The default figure export creates no PDF, SVG, or PNG files. Numerical inputs and measurement units are retained in `06_Results`. Figure regeneration does not insert images into the Word documents. Run regeneration in a working copy to preserve the frozen deliverable.

The main expanded-cohort result contains 511 records, 4,247 later quantified observations, and ten held out sources per approach. LightGBM's source balanced MAE is 1.1143 log units (1.11 rounded); its pooled observation MAE is 1.2162 (1.22 rounded). These are different aggregation choices. The eight comparison approaches include seven fitted model conditions and starting-count persistence.

## Recompute summary results from saved predictions

```
python -B 04_Code/Modeling/InitialConcentration/aggregate_clean_models.py
python -B 04_Code/Modeling/InitialConcentration/summarize_initial_sensitivity.py
python -B 04_Code/Modeling/InitialConcentration/summarize_measured_anchor.py
python -B 04_Code/Modeling/InitialConcentration/organism_results.py
python -B 04_Code/Modeling/InitialConcentration/paired_model_intervals.py
python -B 04_Code/Modeling/InputReliance/summarize_ablations.py
```

The aggregate defaults to the preserved final seed-level prediction files. `--from-training` instead reads newly trained outputs. These are distinct from the historical pilot or candidate fits, which are omitted. Descriptive comparison intervals condition on the selected cohort and fitted models.

## Full model training

Use a **separate working copy of this candidate** with no archived training outputs. From that copy, stage only the clean inputs, split contracts, and record catalog:

```text
python -B stage_private_inputs.py --source-root PATH_TO_PRIVATE_REVIEW_PACKAGE --scope training
```

Then run:

```
python -B 04_Code/train_all.py --full
```

`--full` rejects existing selected-configuration or final-completion markers before running any pilot or fit, so archived results cannot be mistaken for a fresh refit. Do not stage `--scope full` for this workflow, and do not erase delivered files to bypass the check. Resuming a compatible interrupted fit through its controller is a separate operation.

With an empty training destination, the command runs seven numerical pilots and input/split preflight checks, then both controllers, including inner selection and three final seeds per outer fold. Omitting `--full` runs pilots only. New fitted weights and prediction files are written beneath the working copy's `06_Results/Training`; controller bookkeeping and pilots are produced next to its code. The final aggregation creates its `06_Results/Predictions/Dynamic` directory. This is a substantial CPU/GPU job. The separate private review archive includes 105 final weights and matching fit records for saved-model inference; they do not establish an independent refit.

After training, recompute the initial concentration results in this order:

```
python -B 04_Code/Modeling/InitialConcentration/build_initial_count_sensitivity_cohort.py
python -B 04_Code/Modeling/InitialConcentration/run_initial_sensitivity.py
python -B 04_Code/Modeling/NeuralODE/score_initial_node.py
python -B 04_Code/Modeling/InitialConcentration/run_measured_anchor.py
python -B 04_Code/Modeling/InitialConcentration/score_measured_anchor_node.py
python -B 04_Code/Modeling/InitialConcentration/summarize_initial_sensitivity.py
python -B 04_Code/Modeling/InitialConcentration/summarize_measured_anchor.py
```

Inference scripts skip saved fold outputs that already exist. To score new weights, use a working copy and clear only the relevant saved prediction subfolders there. Keep the delivered package as the frozen reference.

## Separate constant temperature analysis

```
python -B 04_Code/Modeling/InputReliance/fit_static_models.py
python -B 04_Code/Modeling/InputReliance/analyze.py
python -B 04_Code/Modeling/InputReliance/ablate.py
python -B 04_Code/Modeling/InputReliance/summarize_ablations.py
```

`save_inputs.py` regenerates the explicit constant temperature input table from the final clean features. The dynamic prediction benchmark and this input reliance analysis use distinct cohorts.

## Raw-data preparation and the official comparator

The separate private review archive retains the original retrieval scripts under its `04_Code/Acquisition` directory as provenance; they are excluded from this candidate, and no new download was performed. This candidate's `04_Code/Modeling/Preprocessing/parse_soap.py` and `parse_dynamic.py` parse a privately staged original archive into tidy tables. The private `05_Data/Processed/Preparation/data/raw_tables` tables retain the original individual count observations and profile ordering; `records_base.parquet` retains the frozen source metadata. The reviewed source mapping and normalized raw tables must also be supplied. The current package does not provide a certified one-command raw-download-to-result pipeline. `corrected_curate.py` generates corrected curated tables from those preserved inputs and refuses to overwrite an existing curation manifest. Rebuild curation only in a disposable copy with that manifest removed. `build_features.py` generates tree and neural preparation inputs; `prepare_clean_training_inputs.py` applies the final non-detection exclusion. Check the resulting input hashes against the supplied contracts before retraining. Regeneration under a different environment can alter serialization hashes even when values match, so investigate a mismatch rather than bypassing the contract.

`build_archive_counts.py` rebuilds the three archive count tables used by Figures 1 and S2. `Modeling/OfficialPredictor/run_official_predictor.py` invokes Node.js to regenerate the six official conditional curves from their frozen scenario parameters. This is a distinct external comparator, not the internally fitted Baranyi backbone.

## Verification scope

The inherited 4 October assembly reports describe figure regeneration, saved-summary recalculation, official-curve reproduction, input and split hashes, source separation, relocated imports, and neural forward passes. Those are dated evidence, not evidence that a later Word edit or figure is unchanged. New checks for this review are recorded in the separate private review archive under its `06_Results/Verification` and `08_Review` directories; consult their dates and stated scope. Those reports are not shipped in this code-only candidate. A complete independent retraining was not performed during either package assembly. This candidate excludes measurements, source correspondence, final weights, and third-party Predictor files; the candidate-root `README.md` explains private staging.
