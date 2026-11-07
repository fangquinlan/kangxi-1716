from common import ROOT, WORK, os
"""Scan exported PNGs for hairlines; sheet of detections (red = would be removed)."""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from PIL import Image
from hairline import find_hairlines, remove_hairlines
from concurrent.futures import ProcessPoolExecutor
OUT = os.environ.get('KX_OUT', os.path.join(ROOT, 'output', '武英殿康熙字典字头'))

def work(p):
    a = np.array(Image.open(p).convert('L'))
    g = (a < 128).astype(np.uint8)
    segs = find_hairlines(g)
    if not segs:
        return None
    g2, n = remove_hairlines(g, segs)
    return (p, segs, n)
if __name__ == '__main__':
    files = []
    for d in ('字头PNG', '重复字头', '部首标题'):
        files += [os.path.join(OUT, d, f) for f in os.listdir(os.path.join(OUT, d))]
    with ProcessPoolExecutor(10) as ex:
        res = [r for r in ex.map(work, files, chunksize=200) if r]
    print('files', len(files), 'with hairlines', len(res))
    json.dump([(p, s, n) for p, s, n in res], open(os.path.join(WORK, 'hairlines.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    random.seed(0)
    sel = res if len(res) <= 96 else random.sample(res, 96)
    tiles = []
    for p, segs, n in sel:
        a = np.array(Image.open(p).convert('L'))
        g = (a < 128).astype(np.uint8)
        g2, _ = remove_hairlines(g, segs)
        v = np.full(g.shape + (3,), 255, np.uint8)
        v[g > 0] = (0, 0, 0)
        v[(g > 0) & (g2 == 0)] = (0, 0, 255)
        tiles.append((v, os.path.basename(p)[:-4][:8] + f' {n}'))
    S, cols = (230, 8)
    rows = max(1, (len(tiles) + cols - 1) // cols)
    sheet = np.full((rows * (S + 16), cols * S, 3), 200, np.uint8)
    for i, (v, lab) in enumerate(tiles):
        rr, cc = divmod(i, cols)
        sc = min(1, (S - 4) / max(v.shape[:2]))
        v2 = cv2.resize(v, (int(v.shape[1] * sc), int(v.shape[0] * sc)), interpolation=cv2.INTER_AREA)
        sheet[rr * (S + 16):rr * (S + 16) + v2.shape[0], cc * S:cc * S + v2.shape[1]] = v2
        cv2.putText(sheet, str(i), (cc * S + 2, rr * (S + 16) + S + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.imwrite(os.path.join(WORK, 'notes', 'hairlines.png'), sheet)
