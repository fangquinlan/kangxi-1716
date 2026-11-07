from common import ROOT, WORK, os
import os, numpy as np, cv2, json
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
OUT = os.path.join(os.environ.get('KX_OUT', os.path.join(ROOT, 'output', '武英殿康熙字典字头')), '字头PNG')

def frag(f):
    a = np.array(Image.open(os.path.join(OUT, f)).convert('L'))
    g = (a < 128).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(g, 8)
    areas = st[1:, 4]
    small = int(((areas >= 6) & (areas < 40)).sum())
    inv = (1 - g).astype(np.uint8)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(inv, 4)
    pin = int(((st2[1:, 4] >= 3) & (st2[1:, 4] < 25)).sum())
    return (f, small, pin)
if __name__ == '__main__':
    fs = os.listdir(OUT)
    with ProcessPoolExecutor(10) as ex:
        res = list(ex.map(frag, fs, chunksize=300))
    sm = np.array([r[1] for r in res])
    pn = np.array([r[2] for r in res])
    print('small comps pct', np.percentile(sm, [50, 90, 99, 99.9]), '>=3:', int((sm >= 3).sum()), '>=5:', int((sm >= 5).sum()))
    print('pinholes pct', np.percentile(pn, [50, 90, 99, 99.9]), '>=5:', int((pn >= 5).sum()), '>=10:', int((pn >= 10).sum()))
    json.dump(sorted(res, key=lambda r: -(r[1] + r[2])), open(os.path.join(WORK, 'frag_all.json'), 'w', encoding='utf-8'), ensure_ascii=False)
