from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'data').is_dir())
sys.path.insert(0,str(_PACKAGE/'code'))
sys.path.insert(0,str(_PACKAGE/'code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
import argparse
import subprocess
parser=argparse.ArgumentParser(description='Run required pilots, then optionally the complete nested source training.')
parser.add_argument('--full',action='store_true',help='Run all fits after pilots; substantial CPU/GPU computation.')
args=parser.parse_args()

if args.full:
    existing=[]
    for family in ['Primary','NeuralODE']:
        runs=RESULTS/'Training'/family/'runs'
        existing.extend(runs.glob('*/fold*/selected.json'))
        existing.extend(runs.glob('*/fold*/final_s*/complete.json'))
    if existing:
        parser.error(
            '--full requires a fresh training destination, but '
            f'{len(existing)} saved selection/completion markers exist in results/Training. '
            'No pilots or fits were started. Use a fresh working copy and place only '
            'the required private training inputs as described in DATA_ACCESS.md. '
            'Run code/train_all.py --full from that working folder. '
            'Keep archived fitted results in a separate private folder. '
            'Direct controller resume is a different operation.'
        )
def call(family,script,*args):
    folder=CODE/'Modeling'/family
    subprocess.run([sys.executable,'-B',str(folder/script),*args],cwd=folder,check=True)
for kind in ['rf','gbm','gru_abs','transformer_abs','backbone','mul']:
    call('Primary','tree.py' if kind in ['rf','gbm'] else 'train.py',kind,'0','0','--seed','0','--pilot')
call('Primary','preflight.py')
call('NeuralODE','train.py','node','0','0','--seed','0','--pilot')
call('NeuralODE','preflight.py')
if args.full:
    call('Primary','controller.py')
    call('NeuralODE','controller.py')
    (RESULTS/'Predictions/Dynamic').mkdir(parents=True,exist_ok=True)
    call('InitialConcentration','aggregate_clean_models.py','--from-training')
