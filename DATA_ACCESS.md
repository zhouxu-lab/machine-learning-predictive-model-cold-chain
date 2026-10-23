# Data and input requirements

Underlying observations originate from [USDA ComBase](https://combase.ars.usda.gov/), subject to its [source terms](https://combase.ars.usda.gov/Disclaimer.aspx). This repository provides model and analysis code. Journal supplementary tables are maintained separately. It does not include the exact archived observations, processed record-level inputs, frozen source assignments, prediction rows, or fitted weights. No public access route to those excluded files is asserted here. Their absence limits exact independent reproduction.

## Inputs for the primary training workflow

Place approved local copies at these paths relative to the repository root:

```text
data/Processed/Training/features.parquet
data/Processed/Training/deep_prepared.pkl
data/Processed/record_catalog.csv
data/Splits_and_Configurations/Primary/contract.json
data/Splits_and_Configurations/Primary/splits.json
data/Splits_and_Configurations/NeuralODE/contract.json
data/Splits_and_Configurations/NeuralODE/splits.json
```

The feature table contains record identifiers, source groups, observation times and targets, starting counts, environmental summaries, and food/organism covariates. The prepared neural input is a tuple of records and category mappings. Records supply temperature grids, covariate vectors, category indices, time-step sizes, starting counts, and observed grid positions and targets. The corresponding preparation and training scripts define the complete field requirements. Deserialize only trusted pickle/model files.

Contracts supply candidate configurations, input hashes, and the split hash. Split files supply the five outer source-held-out train/test partitions and their source-disjoint inner training/validation partitions. The record catalog supplies cohort eligibility and source labels for aggregation. Keep both model families on the same frozen splits. Randomly recreating partitions or downloading current observations does not reconstruct the study inputs.

Verify hashes before training:

```text
python -B code/verify_inputs.py
```

## Additional analyses and saved results

Initial-concentration and constant-temperature analyses require the normalized `data/Processed/Preparation` tables, including their raw/curated tables and features. Neural sensitivity scoring also uses its prepared records. Supply these exact intermediates rather than substituting clean-training rows for a different cohort. Reusing fitted models requires the appropriate `results/Training/Primary` and `results/Training/NeuralODE` outputs.

To check saved benchmark results, provide `censor_safe_matched_points.csv` and `censor_safe_model_metrics.csv` in a local directory and pass that directory to `code/verify_results.py --data`. Reaggregating archived seed predictions additionally requires `results/Predictions/Fitted/Primary` and `NeuralODE`; fresh fitting writes corresponding outputs under `results/Training`.

`data` and `results` are ignored except for their explanatory README files. Keep local inputs and generated outputs out of public commits. The journal supplementary tables describe settings and aggregate results; they cannot substitute for the exact model inputs or fitted outputs. The MIT software license does not extend to third-party observations or journal supplementary materials.
