# Machine learning predictive models for the food cold chain

Model and numerical-analysis code for **Machine learning and hybrid models for predicting microbial growth in cold food supply chains**, by Xu Zhou.

Repository: https://github.com/zhouxu-lab/machine-learning-predictive-model-cold-chain

The code includes Random Forest, LightGBM, GRU, Transformer, a Neural Baranyi ODE, a fitted Baranyi backbone, and its constrained neural hybrid. Evaluation uses experimental sources held out from fitting and model selection. Initial-concentration sensitivity and a separate constant-temperature input-reliance analysis are also included.

## Contents

- `04_Code/Modeling`: input preparation, model definitions, training controllers, and numerical evaluation.
- `04_Code/Shared`: shared features, temperature interpolation, and model components.
- `05_Data` and `06_Results`: placeholders for ignored local inputs and outputs.

Supplementary tables belong to the journal submission materials and are not duplicated here. This repository excludes manuscript figures, plotting code, raw observations, record-level datasets, frozen split membership, fitted weights, and third-party Predictor files.

## Setup and checks

The reference environment is Windows with Python 3.12. Training controllers use Windows process APIs. Install dependencies in a separate environment:

```text
python -m pip install -r 04_Code/requirements.txt
python -B verify_release.py
```

`verify_release.py` uses only the standard library and checks the public file inventory, hashes, Python syntax, and the code-only publication scope. It does not train or validate fitted models. For the reference CUDA 12.8 environment, install the pinned PyTorch wheel from `https://download.pytorch.org/whl/cu128` before the remaining dependencies. Plotting packages are not required.

## Run models and analyses

The private inputs listed in [DATA_ACCESS.md](DATA_ACCESS.md) must be supplied before model execution. In a fresh working copy containing those inputs:

```text
python -B 04_Code/verify_inputs.py
python -B 04_Code/verify_models.py
python -B 04_Code/train_all.py --full
```

The final command performs model selection and fitting and can take substantial CPU/GPU time. [REPRODUCE.md](04_Code/REPRODUCE.md) gives the run order and numerical-analysis commands. Do not reuse an archived fitted-results directory as a fresh training destination.

Exact model execution requires the inputs, source assignments, split contracts, and prediction rows described in DATA_ACCESS.md. Public code alone does not reproduce the study's exact numerical results; a current ComBase download may differ from the retained snapshot. A complete independent refit has not been certified.

## License and citation

Software is provided under the root [MIT license](LICENSE). It does not relicense ComBase observations or journal supplementary materials. See [CITATION.cff](CITATION.cff) for the software citation.
