from common import ROOT, WORK, os
"""Contact sheets of final PNGs: by flag, random sample, or explicit character list."""
import os, sys, json, random
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from sheet import make_sheet
OUT = os.environ.get('KX_OUT', os.environ.get('KX_OUT', os.path.join(ROOT, 'output', '武英殿康熙字典字头')))

def load(p):
    return np.array(Image.open(p).convert('L'))
if __name__ == '__main__':
    mode = sys.argv[1]
    name = sys.argv[2]
    if mode == 'flag':
        res = json.load(open(os.path.join(WORK, 'qa_final.json'), encoding='utf-8'))
        flag = sys.argv[3]
        sel = [r['path'] for r in res if not r.get('empty') and flag in r['flags']]
        random.seed(0)
        random.shuffle(sel)
        print(flag, len(sel))
        sel = sel[:int(sys.argv[4]) if len(sys.argv) > 4 else 96]
    elif mode == 'random':
        d = os.path.join(OUT, '字头PNG')
        fs = sorted(os.listdir(d))
        random.seed(int(sys.argv[3]))
        sel = [os.path.join(d, f) for f in random.sample(fs, int(sys.argv[4]))]
    else:
        sel = [os.path.join(OUT, '字头PNG', ch + '.png') for ch in sys.argv[3]]
        sel = [p for p in sel if os.path.exists(p)]
    imgs = [load(p) for p in sel]
    labels = [os.path.basename(p)[:-4][:10] for p in sel]
    make_sheet(imgs, labels, S=150, cols=12, path=os.path.join(os.path.join(WORK, 'notes'), name))
