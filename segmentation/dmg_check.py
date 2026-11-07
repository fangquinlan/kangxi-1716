"""Show pixels removed from glyph masks by the new line removal (red=frame caps, blue=thin verticals)."""
from pipeline import *
from sheet import make_sheet
specs = sys.argv[1].split(',')   # vol/pno/side
tiles = []
for spec in specs:
    vol, pno, side = spec.split('/')
    pno = int(pno)
    doc = fitz.open(pdf_list()[vol])
    im = load_page(doc, pno)
    norm = normalize_bg(im)
    thr, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink_full = (norm < thr).astype(np.uint8)
    faces = find_faces(ink_full)
    for f in faces: comb_fit(f, ink_full)
    f = [f for f in faces if f['side'] == side][0]
    P = f['pitch']
    for k in range(8):
        strip, y0, xl = extract_strip(norm, f, k)
        ink, dt, _ = prepare_strip(strip, thr)
        ink, rings = remove_rings(ink, P)
        closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        dt = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
        protect = cv2.dilate((dt >= 3.5).astype(np.uint8), np.ones((5, 5), np.uint8))
        v = thin_verticals(ink, P) & (1 - protect)
        cands0, _ = detect_strip(ink, dt, P)
        rc, caps = frame_caps(ink, dt, P, [(c['ty0'], c['ty1']) for c in cands0 if c['thick_area'] >= 200])
        rc = rc & (1 - protect)
        cands, thick = detect_strip(ink, dt, P)
        for c in cands:
            crop, bb = glyph_mask(ink, thick, c, P, dt)
            if crop is None: continue
            bx0, by0, bx1, by1 = bb
            m = np.zeros_like(ink); m[by0:by1 + 1, bx0:bx1 + 1] = crop
            mm = cv2.dilate(m, np.ones((5, 5), np.uint8))
            lost_c = (rc & mm & ink).sum(); lost_v = (v & mm & ink).sum()
            if lost_c + lost_v < 15: continue
            a, b = max(0, by0 - 10), by1 + 11
            vis = np.full((b - a, strip.shape[1], 3), 255, np.uint8)
            vis[ink[a:b] > 0] = (170, 170, 170)
            vis[m[a:b] > 0] = (0, 0, 0)
            vis[(rc[a:b] > 0) & (mm[a:b] > 0)] = (0, 0, 255)
            vis[(v[a:b] > 0) & (mm[a:b] > 0)] = (255, 0, 0)
            tiles.append((vis, f'{vol}/{pno}{side}{k} c{lost_c} v{lost_v}'))
S = 220
cols = 9
rows = (len(tiles) + cols - 1) // cols
sheet = np.full((rows * (S + 20), cols * S, 3), 200, np.uint8)
for i, (v, lab) in enumerate(tiles):
    r, cc = divmod(i, cols)
    sc = min(1, (S - 4) / max(v.shape[:2]))
    v2 = cv2.resize(v, (int(v.shape[1] * sc), int(v.shape[0] * sc)), interpolation=cv2.INTER_AREA)
    sheet[r * (S + 20):r * (S + 20) + v2.shape[0], cc * S:cc * S + v2.shape[1]] = v2
    cv2.putText(sheet, lab, (cc * S + 2, r * (S + 20) + S + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
cv2.imwrite(os.path.join(WORK, 'notes', 'dmg_check.png'), sheet)
print(len(tiles))
