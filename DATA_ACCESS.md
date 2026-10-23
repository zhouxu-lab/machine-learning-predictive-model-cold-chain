# Data access and third-party materials

The original source is [ComBase](https://combase.ars.usda.gov/). Its
[terms and conditions](https://combase.ars.usda.gov/Disclaimer.aspx) apply to
ComBase data and tools. The MIT software license in this candidate does not
license those materials. The terms were checked on 23 October 2026; clause 2.2
addresses reproduction/distribution for personal use, academic or scientific
research, private study, and internal business, and clause 3 reserves the
provider's intellectual property. The source observations and record-level
processed data remain private while the author arranges an access route. Three
author-created supplementary settings and aggregate-result tables are publicly
shared in `Supplementary_Data`; they are distinct from the underlying ComBase
observations. No new data license is asserted by publishing those accompanying
research tables.

The private review package retains the dated source archive, processed inputs,
source-disjoint outer/inner folds, reviewed source correspondence, predictions,
and final fitted weights. Neither downloading current data nor using random
splits substitutes for those frozen inputs. Provide approved reviewer access to
the snapshot under the source terms; the authors must supply that access route.

The code uses a sibling `05_Data` and `06_Results` layout. `stage_private_inputs.py`
copies approved local files to those ignored folders. The `training` scope needs
final clean feature/neural inputs, the record catalog, and both split/contract folders. Full
cohort reconstruction and initial-count sensitivity also require normalized
Preparation tables, `record_catalog.csv`, and saved models. The `full` scope
copies the complete private data/results tree for that workflow.

The official USDA ComBase Predictor is a separate comparator from the internally
fitted Baranyi backbone. To regenerate its six frozen conditional curves, stage
the private `05_Data/Raw/Official_Predictor` folder and
`06_Results/Figure_Source_Data/Official_Predictor` folder, retain their attribution,
and install Node.js. Then run:

```text
python -B 04_Code/Modeling/OfficialPredictor/run_official_predictor.py
```

Those downloaded third-party JavaScript files are excluded from this candidate's
MIT release. The `figures` staging scope includes the returned frozen curves and
scenarios, sufficient for plotting, but does not include third-party source code.
The `full` private scope includes that source code. Neither route changes the
original source-data archive.

ComBase's `-0.01` represents a non-detection code and is not treated as a measured
numeric count in the final model fit. Hypothetical initial concentrations of
0.5–2.0 log units are assumptions, not observed counts or verified detection limits.
