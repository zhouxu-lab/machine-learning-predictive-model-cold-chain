from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'data').is_dir())
sys.path.insert(0,str(_PACKAGE/'code'))
sys.path.insert(0,str(_PACKAGE/'code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
from fit_static_models import data_and_folds
x=data_and_folds()
p=DATA/'Processed/InputReliance';p.mkdir(exist_ok=True)
x.to_parquet(p/'constant_temperature_inputs.parquet',index=False)
print(len(x),x.record_id.nunique(),x.source_key.nunique())
