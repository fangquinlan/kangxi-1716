"""Audit: raw ink removed by strip cleaning (rules, rings, borders, thin verticals) right next to glyph masks."""
import os, sys, random, pickle
sys.path.insert(0, os.path.dirname(__file__))
os.environ['KX_CANDS'] = 'cands_audit'
import numpy as np, cv2
import pipeline
from pipeline import *
from concurrent.futures import ProcessPoolExecutor

def work(a):
    vol, pno = a
    pipeline.process_page((vol, pno))
    rec = pickle.load(open(os.path.join(pipeline.CAND_DIR, vol, f'{pno:04d}.pkl'), 'rb'))
    out = []
    for ci, c in enumerate(rec['cands']):
        h, w = c['crop_shape']
        m = np.unpackbits(c['crop'], axis=1)[:, :w]
        fh, fw = c['forbid_shape']
        fb = np.unpackbits(c['forbid'], axis=1)[:, :fw]
        ox, oy = c['gray_off']
        M = np.zeros((fh, fw), np.uint8)
        M[oy:oy + h, ox:ox + w] = m
        adj = fb & cv2.dilate(M, np.ones((3, 3), np.uint8))
        n = int(adj.sum())
        tile = None
        if n >= 25:
            v = cv2.cvtColor(c['gray'], cv2.COLOR_GRAY2BGR)
            v[M > 0] = (0, 0, 0)
            v[fb > 0] = (0, 0, 255)
            tile = v
        out.append((vol, pno, ci, n, tile))
    return out

if __name__ == '__main__':
    random.seed(3)
    jobs = []
    for vol in ['01', '02', '03', '05', '07', '09', '11', '13']:
        n = fitz.open(pdf_list()[vol]).page_count
        jobs += [(vol, p) for p in random.sample(range(n), 12)]
    res = []
    with ProcessPoolExecutor(10) as ex:
        for o in ex.map(work, jobs):
            res += o
    res.sort(key=lambda r: -r[3])
    print('cands', len(res), 'adj>=25', sum(1 for r in res if r[3] >= 25), 'adj>=60', sum(1 for r in res if r[3] >= 60))
    tiles = [(r[4], f'{r[0]}p{r[1]} c{r[2]} {r[3]}') for r in res if r[4] is not None][:72]
    S = 160; cols = 12
    rows = max(1, (len(tiles) + cols - 1) // cols)
    sheet = np.full((rows * (S + 16), cols * S, 3), 200, np.uint8)
    for i, (v, lab) in enumerate(tiles):
        rr, cc = divmod(i, cols)
        sc = min(1, (S - 4) / max(v.shape[:2]))
        v2 = cv2.resize(v, (int(v.shape[1] * sc), int(v.shape[0] * sc)), interpolation=cv2.INTER_AREA)
        sheet[rr * (S + 16):rr * (S + 16) + v2.shape[0], cc * S:cc * S + v2.shape[1]] = v2
        cv2.putText(sheet, lab, (cc * S + 2, rr * (S + 16) + S + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.imwrite(os.path.join(WORK, 'notes', 'audit_removed.png'), sheet)
