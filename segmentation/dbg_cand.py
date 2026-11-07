"""Debug a single candidate: strip region with ink/thick/glyph mask overlay, plus rendered output."""
from pipeline import *
from render import clean_glyph
vol, pno, side, k, yc = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
doc = fitz.open(pdf_list()[vol])
im = load_page(doc, pno)
norm = normalize_bg(im)
thr, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
ink_full = (norm < thr).astype(np.uint8)
faces = find_faces(ink_full)
for f in faces: comb_fit(f, ink_full)
f = [f for f in faces if f['side'] == side][0]
P = f['pitch']
strip, y0, xl = extract_strip(norm, f, k)
ink, dt, _ = prepare_strip(strip, thr)
ink, rings = remove_rings(ink, P)
ink, _ = remove_lines(ink, P)
closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
dt = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
cands, thick = detect_strip(ink, dt, P)
c = min(cands, key=lambda c: abs((c['ty0'] + c['ty1']) / 2 + y0 - yc))
crop, bb = glyph_mask(ink, thick, c, P, dt)
bx0, by0, bx1, by1 = bb
a, b = max(0, by0 - 200), min(strip.shape[0], by1 + 200)
print('cand rows', c['ty0'] + y0, c['ty1'] + y0, 'all cands', [(cc['ty0'] + y0, cc['ty1'] + y0, cc['thick_area']) for cc in cands if abs(cc['ty0'] + y0 - yc) < 400])
vis = cv2.cvtColor(strip[a:b], cv2.COLOR_GRAY2BGR)
vis[ink[a:b] > 0] = (150, 150, 150)
vis[thick[a:b] > 0] = (0, 0, 255)
m = np.zeros_like(ink); m[by0:by1 + 1, bx0:bx1 + 1] = crop
vis[(m[a:b] > 0) & (thick[a:b] == 0)] = (255, 120, 0)
vis = cv2.resize(vis, (vis.shape[1] * 2, vis.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(os.path.join(WORK, 'notes', 'dbg_cand.png'), vis)
print('P', P, 'bbox', bb, 'y0', y0, 'dt max in strip region', float(dt[a:b].max()))
# thickness stats: dt values along skeleton of non-glyph thin stuff
