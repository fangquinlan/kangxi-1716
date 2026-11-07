"""Prototype citation-box (方框) detection in strips."""
from pipeline import *

def cap_template(w, ls, c=12, bottom=False):
    """Horizontal edge (corners skipped) + two vertical sides of length ls, starting c+6 below the edge."""
    h = c + 6 + ls + 1
    t = np.zeros((h, w + 1), np.float32)
    t[0, c + 2:w - c - 1] = 1
    t[c + 6:c + 6 + ls, 0] = 1
    t[c + 6:c + 6 + ls, w] = 1
    if bottom:
        t = t[::-1]
    return t / t.sum(), h

def find_frames(ink, dt, P, margin=MARGIN, cov_thr=0.80):
    H, W = ink.shape
    D = cv2.dilate(ink, np.ones((3, 3), np.uint8)).astype(np.float32)
    thin_ok = (dt < 3.0)
    res = []
    half = W / 2
    for sub, (xa, xb) in (('L', (margin, half)), ('R', (half, W - margin))):
        for w in range(int(0.40 * P), int(0.53 * P) + 1, 2):
            for bottom in (False, True):
                t, th = cap_template(w, int(0.18 * P), bottom=bottom)
                # allowed left x positions so the frame lies within the sub-column (+-10px slack)
                lo = int(max(0, xa - 10))
                hi = int(min(W - w - 1, xb + 10 - w))
                if hi <= lo:
                    continue
                region = D[:, lo:hi + w + 1]
                r = cv2.matchTemplate(region, t, cv2.TM_CCORR)
                y, x = np.unravel_index(np.argmax(r), r.shape)
                # collect all local peaks above threshold
                ys, xs = np.nonzero(r >= cov_thr)
                for yy, xx in zip(ys, xs):
                    res.append((float(r[yy, xx]), sub, bottom, int(lo + xx), int(yy), w, th))
    # non-max suppression: keep best per (sub, bottom) within 20px in y
    res.sort(key=lambda r: -r[0])
    kept = []
    for r in res:
        if any(k[1] == r[1] and k[2] == r[2] and abs(k[4] - r[4]) < 20 for k in kept):
            continue
        kept.append(r)
    return kept

if __name__ == '__main__':
    vol, pno, side = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    doc = fitz.open(pdf_list()[vol])
    im = load_page(doc, pno)
    norm = normalize_bg(im)
    thr, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink_full = (norm < thr).astype(np.uint8)
    faces = find_faces(ink_full)
    for f in faces: comb_fit(f, ink_full)
    f = [f for f in faces if f['side'] == side][0]
    P = f['pitch']
    strips = []
    t0 = time.time()
    for k in range(8):
        strip, y0, xl = extract_strip(norm, f, k)
        ink, dt, _ = prepare_strip(strip, thr)
        ink, rings = remove_rings(ink, P)
        fr = find_frames(ink, dt, P)
        vis = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)
        for sc, sub, bottom, x, y, w, th in fr:
            col = (0, 0, 255) if not bottom else (255, 0, 0)
            yy = y if not bottom else y + th - 1
            cv2.line(vis, (x, yy), (x + w, yy), col, 3)
            cv2.putText(vis, f'{sc:.2f}', (x, yy - 3 if not bottom else yy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        strips.append(vis)
    print('time', time.time() - t0)
    hmin = min(s.shape[0] for s in strips)
    sheet = np.hstack([s[:hmin] for s in strips[::-1]] if side == 'R' else [s[:hmin] for s in strips[::-1]])
    cv2.imwrite(os.path.join(WORK, 'notes', f'frames_{vol}_{pno}{side}.jpg'), cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2), interpolation=cv2.INTER_AREA))
