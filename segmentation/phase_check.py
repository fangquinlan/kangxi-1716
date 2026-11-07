import os, pickle, numpy as np, cv2, sys
from collections import defaultdict
from align import WORK
from concurrent.futures import ProcessPoolExecutor

def page(args):
    vol, f = args
    rec = pickle.load(open(os.path.join(WORK, 'cands', vol, f), 'rb'))
    out = defaultdict(list)
    for c in rec['cands']:
        h, w = c['crop_shape']
        m = np.unpackbits(c['crop'], axis=1)[:, :w]
        closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        dt = cv2.distanceTransform(np.pad(closed, 2), cv2.DIST_L2, 5)[2:-2, 2:-2]
        ys, xs = np.nonzero(dt >= 4.0)
        if len(xs) < 100:
            continue
        W = int(round(c['P'])) + 48
        off = (c['strip_box'][0] + xs.mean()) - W / 2
        out[(vol, rec['pno'], c['face'])].append(off / c['P'])
    return out

if __name__ == '__main__':
    jobs = [(v, f) for v in sorted(os.listdir(os.path.join(WORK, 'cands'))) for f in sorted(os.listdir(os.path.join(WORK, 'cands', v)))]
    allo = {}
    with ProcessPoolExecutor(10) as ex:
        for o in ex.map(page, jobs, chunksize=8):
            allo.update(o)
    pickle.dump(allo, open(os.path.join(WORK, 'phase_offsets.pkl'), 'wb'))
    flat = np.concatenate([np.array(v) for v in allo.values()])
    print('thick-centroid offset (P units) pct', np.percentile(flat, [1, 5, 25, 50, 75, 95, 99]).round(3))
    sus = []
    for k, v in allo.items():
        v = np.array(v)
        frac = (np.abs(v) > 0.2).mean()
        if len(v) >= 2 and frac >= 0.5:
            sus.append((k, len(v), round(frac, 2), np.round(v, 2).tolist()[:8]))
    print('faces', len(allo), 'suspicious', len(sus))
    for s in sorted(sus): print(s)
