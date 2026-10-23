SUPPLEMENTARY SUMMARY TABLES

Study: Machine learning and hybrid models for predicting microbial growth
in cold food supply chains

These three CSV files contain derived summary results and model settings.
They contain no individual microbial measurements, record identifiers,
source-group membership maps, correspondence records, or fitted model weights.
The original and processed ComBase record archive is kept separately.

FILES AND FIELDS

1. model_settings.csv (7 rows)
   Model: one of the seven fitted model conditions, including the fitted
     Baranyi backbone and its constrained neural hybrid.
   Fixed settings: settings shared across the four candidates for that model.
   Four candidate settings: the prespecified candidate grid c0-c3. Learning
     rates apply to the relevant optimizer; hidden width, leaf/feature settings,
     regularization and hybrid penalties are described in the study methods.
   The starting-count persistence baseline has no fitted parameters and is
   therefore absent from this table.

2. selected_configurations.csv (35 rows: 7 conditions x 5 outer folds)
   model: displayed model name.
   outer_fold: zero-based outer source-held-out fold, 0-4. This is a fold label,
     not a list or map of the records or sources in that fold.
   configuration_index: zero-based selected candidate, 0-3.
   final_epochs: final neural training epoch count selected from the inner
     validations. For the tree models, 1 is bookkeeping, not a training epoch
     or a tree count; their tree counts are in configuration_json.
   configuration_json: machine-readable settings of the selected candidate.
   Neural epoch caps in configuration_json describe the candidate search;
   final_epochs records the final fit's selected epoch count. Each final
   source-held-out condition used seeds 0, 1 and 2, averaged before scoring.

3. organism_performance_full.csv (17 aggregate organism-label rows)
   These are descriptive LightGBM results for the primary low-temperature
   dynamic-food evaluation: 511 records, 4,247 later quantified observations,
   and 10 held-out source groups.
   Organism label: the original grouped or unresolved database label is
     retained; combined labels must not be interpreted as single species.
   Records / sources: number of records and source groups represented in
     that organism group. Source groups overlap across organism rows and
     their counts must not be added to estimate the total number of sources.
   Later counts: number of later quantified observations scored in the group.
   MAE: observation-weighted mean absolute error on the log10 concentration
     scale within the organism group, rounded to two decimal places. This
     is not the overall source-balanced MAE.
   Underpred. >0.5: percentage of scored observations for which prediction
     minus observation is below -0.5 log10 unit, rounded to one decimal place.
   Missed / growth: number of records where predicted growth was missed,
     followed by the number with an observed increase of at least 1 log10
     unit between the starting and final quantified counts. A value 0 / 0
     has no observed growth records and is not a defined missed-growth rate.
   Groups with few records or one source are descriptive; these endpoint
   counts are not product-safety classifications.

WHAT THESE TABLES SUPPORT

Readers can inspect the candidate grids, identify the selected settings for
each model/fold, compare the rounded organism summaries, and check the
reported record/observation coverage. These files accompany the study's
Supplementary Information and analysis code.

WHAT THEY DO NOT REPRODUCE BY THEMSELVES

Rounded aggregate results cannot recreate individual observations or
predictions, exact errors, source-balanced model comparisons, bootstrap
intervals, temperature trajectories, figures or fitted models. Exact numerical
reproduction requires the retained input snapshot, source assignments, frozen
splits, preparation rules, software environment, and fitted or retrained models.
Downloading the current ComBase database alone does not reconstruct those
frozen inputs. No claim of complete independent retraining is made here.

PROVENANCE AND PUBLIC SCOPE

The CSV files are byte-identical copies of the supporting tables reviewed
with the study. Their public release was authorized by the author. This folder
does not redistribute the original or processed microbial record archive,
private correspondence/source adjudications, individual prediction files,
model weights, manuscript, journal-upload documents or author-review material.
Repository release metadata records checksums and applicable licence scope.
