import os, sys, pickle, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from align import WORK

def page(args):
    vol, f = args
    rec = pickle.load(open(os.path.join(WORK, 'cands', vol, f), 'rb'))
    out = []
    for ci, c in enumerate(rec['cands']):
        h, w = c['crop_shape']
        m = np.unpackbits(c['crop'], axis=1)[:, :w] > 0
        ox, oy = c['gray_off']
        g = c['gray'][oy:oy + h, ox:ox + w]
        if m.sum() < 50:
            continue
        out.append((vol, rec['pno'], ci, float(np.median(g[m])), float(rec['thr'])))
    return out

if __name__ == '__main__':
    jobs = [(v, f) for v in sorted(os.listdir(os.path.join(WORK, 'cands'))) for f in sorted(os.listdir(os.path.join(WORK, 'cands', v)))]
    res = []
    with ProcessPoolExecutor(10) as ex:
        for o in ex.map(page, jobs, chunksize=8):
            res += o
    med = np.array([r[3] for r in res])
    print('cands', len(res), 'median ink gray pct', np.percentile(med, [50, 90, 95, 99, 99.5, 99.9]).round(1))
    for t in (80, 90, 100, 110):
        print('ink median >', t, int((med > t).sum()))
    pickle.dump(res, open(os.path.join(WORK, 'faint_stats.pkl'), 'wb'))
