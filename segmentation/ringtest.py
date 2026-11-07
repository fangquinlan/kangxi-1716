from pipeline import *
import time

def ring_detect(ink, P, debug=False):
    """Find circled marks in a strip via annulus template correlation (half resolution)."""
    H, W = ink.shape
    small = cv2.resize(ink.astype(np.float32), (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    small = (small > 0.2).astype(np.float32)
    small = cv2.dilate(small, np.ones((2, 2), np.uint8))
    found = []
    best_map = None
    for r in np.arange(0.34 * P / 2, 0.45 * P / 2, 1.5):
        R = int(np.ceil(r + 4))
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        d = np.sqrt(yy ** 2 + xx ** 2)
        ann = ((d >= r - 1.5) & (d <= r + 1.5)).astype(np.float32)
        res = cv2.matchTemplate(small, ann, cv2.TM_CCORR) / ann.sum()
        # restrict centers to near column center
        if best_map is None:
            best_map = np.full(res.shape, -1.0, np.float32)
            rad_map = np.zeros(res.shape, np.float32)
        h2 = min(best_map.shape[0], res.shape[0]); w2 = min(best_map.shape[1], res.shape[1])
        # offset alignment: res index (y, x) -> center (y + R, x + R)
        upd = res[:h2, :w2] > best_map[:h2, :w2]
        best_map[:h2, :w2][upd] = res[:h2, :w2][upd]
        rad_map[:h2, :w2][upd] = r
    return best_map, rad_map

if __name__ == '__main__':
    vol, pno = sys.argv[1], int(sys.argv[2])
    doc = fitz.open(pdf_list()[vol])
    im = load_page(doc, pno)
    norm = normalize_bg(im)
    thr, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink_full = (norm < thr).astype(np.uint8)
    faces = find_faces(ink_full)
    for f in faces: comb_fit(f, ink_full)
    t = time.time()
    for f in faces:
        P = f['pitch']
        for k in range(8):
            strip, y0, xl = extract_strip(norm, f, k)
            ink, dt, _ = prepare_strip(strip, thr)
            # use raw ink (rings are thin)
            H, W = ink.shape
            small = cv2.resize(ink.astype(np.float32), (W // 2, H // 2), interpolation=cv2.INTER_AREA)
            small = (small > 0.2).astype(np.float32)
            small = cv2.dilate(small, np.ones((3, 3), np.uint8))
            best = None
            rows = []
            for r in np.arange(0.34 * P / 2, 0.46 * P / 2, 1.5):
                R = int(np.ceil(r + 3))
                yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
                d = np.sqrt(yy ** 2 + xx ** 2)
                ann = ((d >= r - 1.0) & (d <= r + 1.0)).astype(np.float32)
                res = cv2.matchTemplate(small, ann, cv2.TM_CCORR) / ann.sum()
                # column centre region
                cxs = W // 4 - R
                lo, hi = max(0, cxs - 8), min(res.shape[1], cxs + 9)
                sub = res[:, lo:hi]
                m = sub.max(1)
                rows.append((r, R, m, lo + sub.argmax(1)))
            # combine: max over r
            M = np.max(np.stack([m for _, _, m, _ in rows if len(m) == len(rows[-1][2])]), 0) if rows else None
            # peaks
            for (r, R, m, ax) in rows:
                pass
            allm = np.full(H // 2, 0.0)
            arg = np.zeros(H // 2, int)
            for idx, (r, R, m, ax) in enumerate(rows):
                yc = np.arange(len(m)) + R
                ok = yc < len(allm)
                better = m[ok] > allm[yc[ok]]
                allm[yc[ok][better]] = m[ok][better]
                arg[yc[ok][better]] = idx
            peaks = [y for y in range(1, len(allm) - 1) if allm[y] >= 0.6 and allm[y] == allm[max(0, y - 20):y + 21].max()]
            for y in peaks:
                print(f['side'], k, 'y', y * 2 + y0, 'score', round(allm[y], 2), 'r', round(rows[arg[y]][0] * 2, 1))
    print('time', time.time() - t)
