from common import ROOT, WORK, os
"""QA over exported PNGs: format checks + flags for residual thin straight lines near the glyph border."""
import os, sys, csv, json, random
import numpy as np, cv2
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
OUT = os.environ.get('KX_OUT', os.environ.get('KX_OUT', os.path.join(ROOT, 'output', '武英殿康熙字典字头')))

def check(path):
    im = Image.open(path)
    mode = im.mode
    a = np.array(im.convert('L'))
    vals = set(np.unique(a).tolist())
    g = (a < 128).astype(np.uint8)
    ys, xs = np.nonzero(g)
    if len(ys) == 0:
        return dict(path=path, empty=True)
    m = dict(top=int(ys.min()), bot=int(a.shape[0] - 1 - ys.max()), left=int(xs.min()), right=int(a.shape[1] - 1 - xs.max()))
    G = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    H, W = G.shape
    flags = []
    closed = cv2.morphologyEx(G, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    dt = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
    thin = (G > 0) & (dt <= 2.0)
    hl = cv2.morphologyEx(thin.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, max(25, int(0.22 * W))), np.uint8))
    if hl[int(0.82 * H):].any():
        flags.append('hline_bottom')
    if hl[:int(0.18 * H)].any():
        flags.append('hline_top')
    vl = cv2.morphologyEx(thin.astype(np.uint8), cv2.MORPH_OPEN, np.ones((max(25, int(0.25 * H)), 1), np.uint8))
    if vl[:, :int(0.15 * W)].any():
        flags.append('vline_left')
    if vl[:, int(0.85 * W):].any():
        flags.append('vline_right')
    asp = W / H
    if asp > 1.6 or asp < 0.45:
        flags.append('aspect')
    n, lab, st, _ = cv2.connectedComponentsWithStats(G, 8)
    small = int((st[1:, 4] < 30).sum())
    if small >= 6:
        flags.append('specks')
    return dict(path=path, mode=mode, vals=sorted(vals), margins=m, W=W, H=H, flags=flags)

def main():
    files = []
    for d in ('字头PNG', '重复字头', '部首标题'):
        p = os.path.join(OUT, d)
        files += [os.path.join(p, f) for f in os.listdir(p)]
    with ProcessPoolExecutor(10) as ex:
        res = list(ex.map(check, files, chunksize=200))
    errs = [r for r in res if r.get('empty') or r['mode'] != '1' or (not set(r['vals']) <= {0, 255}) or (min(r['margins'].values()) != 10)]
    from collections import Counter
    fc = Counter((f for r in res if not r.get('empty') for f in r['flags']))
    print('files', len(res), 'format/margin errors', len(errs), 'flags', dict(fc))
    json.dump(res, open(os.path.join(os.path.join(WORK), 'qa_final.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    return res
if __name__ == '__main__':
    main()
