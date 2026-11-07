"""Apply font-guided cleanup to the user's regression list + a random sample; visualize removed pixels."""
import os, sys, pickle, random, time
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('KX_CANDS', 'cands_test')
import numpy as np, cv2
import pipeline
from align import WORK
from render import clean_glyph
from fontclean import register, extraneous

def vis_tile(img, ch, label):
    g = (img < 128).astype(np.uint8)
    fimg, cc = register(g, ch)
    rem, st = extraneous(g, fimg) if fimg is not None else (np.zeros(g.shape, bool), dict(removed=0))
    v = np.full(g.shape + (3,), 255, np.uint8)
    if fimg is not None:
        v[fimg > 0.5] = (255, 225, 200)          # font support (light blue)
    v[g > 0] = (0, 0, 0)
    v[rem] = (0, 0, 255)                         # removed (red)
    return v, f"{label} cc{cc:.2f} r{st['removed']}", st['removed']

tiles = []
LISTS = {'02': '嗸㗙夰𠯚𡕽嗿𡣐嗅𠲺𡁊𡣊㜎', '01': '㔑𠆤𠘶𠩞价𠐻𠓆决儚勱㒵𠨰𠕝㐡劈𠟈𠑺勗勳𠨓'}
t = time.time()
for vol, chars in LISTS.items():
    A = pickle.load(open(os.path.join(WORK, f'align_{vol}.pkl'), 'rb'))
    for ch in chars:
        r = max([r for r in A['res'] if r['ch'] == ch], key=lambda r: r['sim'])
        p = os.path.join(pipeline.CAND_DIR, vol, f"{r['pno']:04d}.pkl")
        if not os.path.exists(p):
            pipeline.process_page((vol, r['pno']))
        rec = pickle.load(open(p, 'rb'))
        c = min(rec['cands'], key=lambda c: (c['face'] != r['face']) * 1e6 + (c['col'] != r['col']) * 1e5 + abs(c['yc'] - r['yc']))
        tiles.append(vis_tile(clean_glyph(c, rec['thr']), ch, f'{vol}p{r["pno"]}'))
print('regression', time.time() - t)
S = 180; cols = 11
rows = (len(tiles) + cols - 1) // cols
sheet = np.full((rows * (S + 18), cols * S, 3), 200, np.uint8)
for i, (v, lab, n) in enumerate(tiles):
    rr, cc = divmod(i, cols)
    sc = min(1, (S - 4) / max(v.shape[:2]))
    v2 = cv2.resize(v, (int(v.shape[1] * sc), int(v.shape[0] * sc)), interpolation=cv2.INTER_AREA)
    sheet[rr * (S + 18):rr * (S + 18) + v2.shape[0], cc * S:cc * S + v2.shape[1]] = v2
    cv2.putText(sheet, lab, (cc * S + 2, rr * (S + 18) + S + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
cv2.imwrite(os.path.join(WORK, 'notes', 'fontclean_regression.png'), sheet)
