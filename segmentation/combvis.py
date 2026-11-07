from geom import *
import sys
import pickle
from sheet import make_sheet
items, labels = [], []
for spec in sys.argv[1].split(','):
    vol, pno, side = spec.split('/')
    pno = int(pno)
    rec = pickle.load(open(os.path.join(WORK, 'cands', vol, f'{pno:04d}.pkl'), 'rb'))
    old = [f for f in rec['faces'] if f['side'] == side][0]
    doc = fitz.open(pdf_list()[vol])
    im = load_page(doc, pno)
    norm = normalize_bg(im)
    thr, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (norm < thr).astype(np.uint8)
    faces = find_faces(ink)
    new = [f for f in faces if f['side'] == side][0]
    comb_fit(new, ink)
    vis = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    sgn = 1 if side == 'R' else -1
    for f, col, dx in ((old, (0, 0, 255), -3), (new, (0, 170, 0), 3)):
        P = f['pitch']
        for u in [f['rules_u'][0] - P] + f['rules_u']:
            p0 = (int(gutter_x(f, f['ytop']) + sgn * u + dx), int(f['ytop']))
            p1 = (int(gutter_x(f, f['ybot']) + sgn * u + dx), int(f['ybot']))
            cv2.line(vis, p0, p1, col, 4)
    xg = int(gutter_x(new, 1600))
    x0, x1 = (xg - 80, xg + 1750) if side == 'R' else (xg - 1750, xg + 80)
    x0 = max(0, x0); x1 = min(vis.shape[1], x1)
    crop = vis[int(new['ytop']) - 40:int(new['ybot']) + 40, x0:x1]
    items.append(cv2.cvtColor(cv2.resize(crop, (crop.shape[1] // 4, crop.shape[0] // 4), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    labels.append(spec)
# color sheet
S = 470
cols = 4
rows = (len(items) + cols - 1) // cols
sheet = np.full((rows * (S + 24), cols * S, 3), 200, np.uint8)
for i, im in enumerate(items):
    r, c = divmod(i, cols)
    h, w = im.shape[:2]
    sc = min(1, (S - 4) / max(h, w))
    im2 = cv2.resize(im, (int(w * sc), int(h * sc)))
    sheet[r * (S + 24):r * (S + 24) + im2.shape[0], c * S:c * S + im2.shape[1]] = im2
    cv2.putText(sheet, labels[i], (c * S + 4, r * (S + 24) + S + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
cv2.imwrite(os.path.join(WORK, 'notes', 'combvis.png'), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
