"""Run font-guided cleanup over a random sample of matched candidates (current cands/), show everything removed."""
import os, sys, pickle, random, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from align import WORK, load_cands
from render import clean_glyph
from fontclean import register, extraneous
from concurrent.futures import ProcessPoolExecutor

def work(args):
    vol, pno, items = args
    rec = pickle.load(open(os.path.join(WORK, 'cands', vol, f'{pno:04d}.pkl'), 'rb'))
    out = []
    for ci, ch, sim in items:
        img = clean_glyph(rec['cands'][ci], rec['thr'])
        g = (img < 128).astype(np.uint8)
        fimg, cc = register(g, ch)
        if fimg is None or cc < 0.85:
            out.append((vol, pno, ci, ch, sim, cc, 0, None))
            continue
        rem, st = extraneous(g, fimg)
        tile = None
        if st['removed']:
            v = np.full(g.shape + (3,), 255, np.uint8)
            v[g > 0] = (0, 0, 0)
            v[rem] = (0, 0, 255)
            tile = v
        out.append((vol, pno, ci, ch, sim, cc, st['removed'], tile))
    return out

if __name__ == '__main__':
    random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    N = int(sys.argv[1])
    jobs = {}
    for vol in ['01', '02', '04', '06', '08', '10', '12', '13']:
        A = pickle.load(open(os.path.join(WORK, f'align_{vol}.pkl'), 'rb'))
        ms = [r for r in A['res'] if r['match'] >= 0]
        for r in random.sample(ms, N // 8):
            jobs.setdefault((vol, r['pno']), []).append((r['ci'], r['ch'], r['sim']))
    t = time.time()
    res = []
    with ProcessPoolExecutor(10) as ex:
        for o in ex.map(work, [(v, p, it) for (v, p), it in jobs.items()]):
            res += o
    print('glyphs', len(res), 'time', round(time.time() - t, 1))
    rem = [r for r in res if r[6] > 0]
    print('with removal', len(rem), 'skipped(cc<0.5)', sum(1 for r in res if r[5] < 0.5))
    tiles = [(r[7], f'{r[0]}p{r[1]} {r[6]}') for r in sorted(rem, key=lambda r: -r[6])]
    S = 150; cols = 12
    rows = max(1, (len(tiles) + cols - 1) // cols)
    sheet = np.full((rows * (S + 16), cols * S, 3), 200, np.uint8)
    for i, (v, lab) in enumerate(tiles):
        rr, cc = divmod(i, cols)
        sc = min(1, (S - 4) / max(v.shape[:2]))
        v2 = cv2.resize(v, (int(v.shape[1] * sc), int(v.shape[0] * sc)), interpolation=cv2.INTER_AREA)
        sheet[rr * (S + 16):rr * (S + 16) + v2.shape[0], cc * S:cc * S + v2.shape[1]] = v2
        cv2.putText(sheet, lab, (cc * S + 2, rr * (S + 16) + S + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.imwrite(os.path.join(WORK, 'notes', 'fontclean_sample.png'), sheet)
