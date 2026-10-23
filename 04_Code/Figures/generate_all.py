from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'05_Data').is_dir())
sys.path.insert(0,str(_PACKAGE/'04_Code'))
sys.path.insert(0,str(_PACKAGE/'04_Code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
import subprocess
for name in ['1','2','3','4','5','s1','s2']:
    subprocess.run([sys.executable,str(CODE/'Figures'/('figure_'+name+'.py'))],check=True)
