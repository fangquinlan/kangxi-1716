"""Side-by-side: original scan region (from manifest coordinates) and the exported PNG, for given characters."""
import csv, os, sys
import numpy as np, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from common import pdf_list, load_page
import fitz
OUT = os.environ.get('KX_OUT', os.path.join(ROOT, 'output', '武英殿康熙字典字头'))
rows = {r['文件']: r for r in csv.DictReader(open(os.path.join(OUT, '字头清单.csv'), encoding='utf-8-sig'))}
tiles = []
docs = {}
for ch in sys.argv[1]:
    f = f'字头PNG/{ch}.png'
    r = rows[f]
    print(ch, r['册'], 'p', r['PDF页'], r['面'], r['行(自右)'], 'sim', r['字形相似度'], r['来源'], r['字形补全'], r['字体清理像素'])
    a = np.array(Image.open(os.path.join(OUT, f)).convert('L'))
    if r['册'] not in docs:
        docs[r['册']] = fitz.open(pdf_list()[r['册']])
    im = load_page(docs[r['册']], int(r['PDF页']) - 1)
    x0, y0, x1, y1 = (int(r['页面x0']), int(r['页面y0']), int(r['页面x1']), int(r['页面y1']))
    g = im[max(0, y0 - 30):y1 + 30, max(0, x0 - 40):x1 + 40]
    h = max(a.shape[0], g.shape[0])
    pad = lambda z: np.vstack([z, np.full((h - z.shape[0], z.shape[1]), 255, np.uint8)])
    tiles.append(np.hstack([pad(g), np.full((h, 8), 128, np.uint8), pad(a), np.full((h, 20), 128, np.uint8)]))
H = max((t.shape[0] for t in tiles))
out = np.hstack([np.vstack([t, np.full((H - t.shape[0], t.shape[1]), 255, np.uint8)]) for t in tiles])
sc = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
cv2.imwrite(os.path.join(os.path.join(WORK, 'notes'), sys.argv[2]), cv2.resize(out, (int(out.shape[1] * sc), int(out.shape[0] * sc)), interpolation=cv2.INTER_AREA))
