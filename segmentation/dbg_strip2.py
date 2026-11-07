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
ink0, dt, _ = prepare_strip(strip, thr)
ink, rings = remove_rings(ink0, P)
print('rings', [(round(cy + y0), round(r), round(sc, 2)) for cx, cy, r, sc in rings])
closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
dt = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
cands, thick = detect_strip(ink, dt, P)
print('cands', [(c['ty0'] + y0, c['ty1'] + y0, c['thick_area']) for c in cands])
vis = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)
vis[(ink0 > 0) & (ink == 0)] = (255, 0, 255)   # removed by rings
vis[ink > 0] = (120, 120, 120)
vis[thick > 0] = (0, 0, 255)
for c in cands:
    crop, bb = glyph_mask(ink, thick, c, P, dt)
    if crop is None: continue
    bx0, by0, bx1, by1 = bb
    m = np.zeros_like(ink); m[by0:by1 + 1, bx0:bx1 + 1] = crop
    vis[(m > 0) & (thick == 0)] = (255, 140, 0)
    cv2.rectangle(vis, (bx0, by0), (bx1, by1), (0, 160, 0), 2)
for cx, cy, r, sc in rings:
    cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 200, 255), 1)
cv2.imwrite(os.path.join(WORK, 'notes', f'dbgstrip_{vol}_{pno}_{side}{k}.png'), vis)
