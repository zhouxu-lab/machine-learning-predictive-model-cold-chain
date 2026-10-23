from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'05_Data').is_dir())
sys.path.insert(0,str(_PACKAGE/'04_Code'))
sys.path.insert(0,str(_PACKAGE/'04_Code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
import io
import json
from PIL import Image
import matplotlib.pyplot as plt
from audit_panel_alignment import require_matplotlib_panel_alignment

def export(fig,name,axes=None,colorbars=(),tight=True):
    fig.canvas.draw()
    qa=RESULTS/'Verification';qa.mkdir(exist_ok=True)
    if axes is not None:
        require_matplotlib_panel_alignment(fig,axes=axes,exclude_axes=list(colorbars),strict=True,json_out=str(qa/(name+'.alignment.json')))
    buffer=io.BytesIO()
    fig.savefig(buffer,format='png',dpi=600,bbox_inches='tight' if tight else None,facecolor='white')
    buffer.seek(0);im=Image.open(buffer).convert('RGB')
    scale=min(1,(17.1/2.54*600)/im.width,(23.3/2.54*600)/im.height)
    if scale<1:im=im.resize((int(im.width*scale),int(im.height*scale)),Image.Resampling.LANCZOS)
    out=PACKAGE_ROOT/'03_Figures'/(name+'.tiff')
    im.save(out,dpi=(600,600),compression='tiff_lzw')
    plt.close(fig)
    print(out.name,im.size,flush=True)
