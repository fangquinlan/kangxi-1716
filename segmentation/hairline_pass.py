from common import ROOT, WORK, os
"""Post-pass over exported PNGs: remove straight hairlines (scratches/cracks) that lie mostly outside the
registered font outline. Writes a review sheet of every change and records pixel counts per file."""
import os, sys, csv, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from PIL import Image
from hairline import find_hairlines, remove_hairlines, thin_mask
from fontclean import register
from render import save_1bit
from concurrent.futures import ProcessPoolExecutor
OUT = os.environ.get('KX_OUT', os.environ.get('KX_OUT', os.path.join(ROOT, 'output', '武英殿康熙字典字头')))
DRY = os.environ.get('KX_DRY') == '1'
_E9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

def accept(g, segs, fimg, cc):
    if fimg is None or cc < 0.85:
        H, W = g.shape
        return [s for s in segs if max(abs(s[2] - s[0]), abs(s[3] - s[1])) >= 0.35 * max(H, W)]
    support = cv2.dilate((fimg > 0.5).astype(np.uint8), _E9)
    thin = thin_mask(g)
    out = []
    for s in segs:
        x1, y1, x2, y2 = s[:4]
        n = int(max(abs(x2 - x1), abs(y2 - y1))) + 1
        xs = np.round(np.linspace(x1, x2, n)).astype(int)
        ys = np.round(np.linspace(y1, y2, n)).astype(int)
        pts = (g[ys, xs] > 0) & (thin[ys, xs] > 0)
        if pts.sum() < 10:
            continue
        outside = (support[ys, xs][pts] == 0).mean()
        if outside >= 0.6:
            out.append(s)
    return out

def work(item):
    f, ch = item
    p = os.path.join(OUT, f)
    a = np.array(Image.open(p).convert('L'))
    g = (a < 128).astype(np.uint8)
    segs = find_hairlines(g)
    if not segs:
        return (f, 0, None)
    fimg, cc = register(g, ch)
    segs = accept(g, segs, fimg, cc)
    if not segs:
        return (f, 0, None)
    g2, n = remove_hairlines(g, segs)
    if n == 0:
        return (f, 0, None)
    ys, xs = np.nonzero(g2)
    g3 = g2[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    img = np.full((g3.shape[0] + 20, g3.shape[1] + 20), 255, np.uint8)
    img[10:-10, 10:-10][g3 > 0] = 0
    if not DRY:
        save_1bit(img, p)
    v = np.full(g.shape + (3,), 255, np.uint8)
    v[g > 0] = (0, 0, 0)
    v[(g > 0) & (g2 == 0)] = (0, 0, 255)
    return (f, n, v)
if __name__ == '__main__':
    rows = list(csv.DictReader(open(os.path.join(OUT, '字头清单.csv'), encoding='utf-8-sig')))
    items = [(r['文件'], r['字']) for r in rows]
    with ProcessPoolExecutor(10) as ex:
        res = list(ex.map(work, items, chunksize=100))
    changed = [(f, n, v) for f, n, v in res if n]
    print('files', len(res), 'changed', len(changed), 'dry' if DRY else '')
    json.dump({f: n for f, n, v in changed}, open(os.path.join(os.path.join(WORK), 'hairline_changes.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    S, cols = (200, 9)
    rws = max(1, (len(changed) + cols - 1) // cols)
    sheet = np.full((rws * (S + 16), cols * S, 3), 200, np.uint8)
    for i, (f, n, v) in enumerate(changed):
        rr, c = divmod(i, cols)
        sc = min(1, (S - 4) / max(v.shape[:2]))
        v2 = cv2.resize(v, (int(v.shape[1] * sc), int(v.shape[0] * sc)), interpolation=cv2.INTER_AREA)
        sheet[rr * (S + 16):rr * (S + 16) + v2.shape[0], c * S:c * S + v2.shape[1]] = v2
        cv2.putText(sheet, f'{i} {n}', (c * S + 2, rr * (S + 16) + S + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
    cv2.imwrite(os.path.join(os.path.join(WORK, 'notes'), 'hairline_pass.png'), sheet)
