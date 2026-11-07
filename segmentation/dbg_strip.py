from pipeline import *
import detect as D
vol, pno, side, k = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
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
H, W = ink.shape
# ring scores along the strip
small = cv2.resize(ink.astype(np.float32), (W // 2, H // 2), interpolation=cv2.INTER_AREA)
small = (small > 0.2).astype(np.float32)
small = cv2.dilate(small, np.ones((3, 3), np.uint8))
best = np.zeros(H // 2)
for r in np.arange(0.345 * P / 2, 0.415 * P / 2, 1.5):
    ann, R = D._annulus(r)
    res = cv2.matchTemplate(small, ann, cv2.TM_CCORR)
    cxs = W // 4 - R
    lo, hi = max(0, cxs - 8), min(res.shape[1], cxs + 9)
    m = res[:, lo:hi].max(1)
    yc = np.arange(len(m)) + R
    ok = yc < len(best)
    best[yc[ok]] = np.maximum(best[yc[ok]], m[ok])
pk = [(y * 2 + y0, round(float(best[y]), 2)) for y in range(1, len(best) - 1) if best[y] > 0.7 and best[y] == best[max(0, y - 15):y + 16].max()]
print('ring peaks', pk)
rings = find_rings(ink, P)
print('rings', [(round(cy + y0), round(sc, 2)) for cx, cy, r, sc in rings])
ink2, rings = remove_rings(ink, P)
closed = cv2.morphologyEx(ink2, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
dt2 = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
thick = (dt2 >= DT_THICK).astype(np.uint8)
thick[:, :MARGIN + 2] = 0
thick[:, W - MARGIN - 2:] = 0
n, lab, st, _ = cv2.connectedComponentsWithStats(thick, 8)
small_ids = np.nonzero(st[:, 4] < 30)[0]
small_ids = small_ids[small_ids > 0]
thick[np.isin(lab, small_ids)] = 0
fc = fine_clusters(thick, 0.12 * P)
print('P', P, 'fine', [(a + y0, b + y0, int(thick[a:b + 1].sum())) for a, b in fc])
mc = merge_clusters(fc, P)
print('merged', [(a + y0, b + y0, round((b - a) / P, 2)) for a, b in mc])
