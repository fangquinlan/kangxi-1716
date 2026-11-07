from common import ROOT, WORK, os
"""Post-pass: remove detached thin fragments (box-edge corners, rule pieces) lying entirely outside the
registered font outline near the glyph border. Review sheet of every change."""
import os, sys, csv, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from PIL import Image
from fontclean import register
from render import save_1bit
from concurrent.futures import ProcessPoolExecutor
OUT = os.environ.get('KX_OUT', os.environ.get('KX_OUT', os.path.join(ROOT, 'output', '武英殿康熙字典字头')))
DRY = os.environ.get('KX_DRY') == '1'
_E13 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))

def work(item):
    f, ch = item
    p = os.path.join(OUT, f)
    a = np.array(Image.open(p).convert('L'))
    g = (a < 128).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(g, 8)
    if n <= 2:
        return (f, 0, None, None)
    fimg, cc = register(g, ch)
    if fimg is None or cc < 0.8:
        return (f, 0, None, cc)
    support = cv2.dilate((fimg > 0.5).astype(np.uint8), _E13) > 0
    dt = cv2.distanceTransform(np.pad(g, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    H, W = g.shape
    main = 1 + int(np.argmax(st[1:, 4]))
    rem = np.zeros(g.shape, bool)
    for i in range(1, n):
        if i == main:
            continue
        x, y, w, h, ar = st[i]
        comp = lab == i
        if ar < 30 or dt[comp].max() > 2.6:
            continue
        if (comp & support).sum() > 0.05 * ar:
            continue
        rest = (g > 0) & ~comp
        dist = cv2.distanceTransform((~rest).astype(np.uint8), cv2.DIST_L2, 3)
        if dist[comp].min() <= 6:
            continue
        cy, cx = (y + h / 2, x + w / 2)
        border = cy > 0.7 * H or cy < 0.3 * H or cx > 0.7 * W or (cx < 0.3 * W)
        if not border or max(w, h) < 15:
            continue
        rem |= comp
    k = int(rem.sum())
    if k == 0:
        return (f, 0, None, cc)
    g2 = g.copy()
    g2[rem] = 0
    ys, xs = np.nonzero(g2)
    g3 = g2[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    img = np.full((g3.shape[0] + 20, g3.shape[1] + 20), 255, np.uint8)
    img[10:-10, 10:-10][g3 > 0] = 0
    if not DRY:
        save_1bit(img, p)
    v = np.full(g.shape + (3,), 255, np.uint8)
    v[support] = (255, 230, 210)
    v[g > 0] = (0, 0, 0)
    v[rem] = (0, 0, 255)
    return (f, k, v, cc)
if __name__ == '__main__':
    rows = list(csv.DictReader(open(os.path.join(OUT, '字头清单.csv'), encoding='utf-8-sig')))
    items = [(r['文件'], r['字']) for r in rows]
    with ProcessPoolExecutor(10) as ex:
        res = list(ex.map(work, items, chunksize=100))
    changed = [(f, k, v) for f, k, v, cc in res if k]
    print('files', len(res), 'changed', len(changed), 'dry' if DRY else '', 'low-cc skipped', sum((1 for r in res if r[3] is not None and r[3] < 0.8)))
    json.dump({f: k for f, k, v in changed}, open(os.path.join(os.path.join(WORK), 'detached_changes.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    S, cols = (170, 10)
    rws = max(1, (len(changed) + cols - 1) // cols)
    sheet = np.full((rws * (S + 16), cols * S, 3), 200, np.uint8)
    for i, (f, k, v) in enumerate(changed[:300]):
        rr, c = divmod(i, cols)
        sc = min(1, (S - 4) / max(v.shape[:2]))
        v2 = cv2.resize(v, (int(v.shape[1] * sc), int(v.shape[0] * sc)), interpolation=cv2.INTER_AREA)
        sheet[rr * (S + 16):rr * (S + 16) + v2.shape[0], c * S:c * S + v2.shape[1]] = v2
        cv2.putText(sheet, f'{i} {k}', (c * S + 2, rr * (S + 16) + S + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
    cv2.imwrite(os.path.join(os.path.join(WORK, 'notes'), 'detached_pass.png'), sheet)
