from pipeline import *
import detect as D
from concurrent.futures import ProcessPoolExecutor

def ann(r, w=1.0):
    R = int(np.ceil(r + 8))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    d = np.sqrt(yy ** 2 + xx ** 2)
    ring = ((d >= r - w) & (d <= r + w)).astype(np.float32)
    inner = ((d >= r - 5 - w) & (d <= r - 5 + w)).astype(np.float32)
    outer = ((d >= r + 4.5 - w) & (d <= r + 4.5 + w)).astype(np.float32)
    return ring / ring.sum(), inner / inner.sum(), outer / outer.sum(), R

def page_scores(a):
    vol, pno, items = a
    doc = fitz.open(pdf_list()[vol])
    im = load_page(doc, pno)
    norm = normalize_bg(im)
    thr, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink_full = (norm < thr).astype(np.uint8)
    faces = find_faces(ink_full)
    for f in faces: comb_fit(f, ink_full)
    out = []
    for side, k, y, s in items:
        f = [f for f in faces if f['side'] == side][0]
        P = f['pitch']
        strip, y0, xl = extract_strip(norm, f, k)
        ink, dt, _ = prepare_strip(strip, thr)
        H, W = ink.shape
        small = cv2.resize(ink.astype(np.float32), (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        small = (small > 0.2).astype(np.float32)
        small = cv2.dilate(small, np.ones((3, 3), np.uint8))
        yy = (y - y0) // 2
        best = None
        for r in np.arange(0.345 * P / 2, 0.415 * P / 2, 1.5):
            ring, inner, outer, R = ann(r)
            for cy in range(yy - 4, yy + 5):
                for cx in range(W // 4 - 8, W // 4 + 9):
                    if cy - R < 0 or cy + R + 1 > small.shape[0] or cx - R < 0 or cx + R + 1 > small.shape[1]:
                        continue
                    patch = small[cy - R:cy + R + 1, cx - R:cx + R + 1]
                    c0 = (patch * ring).sum(); ci = (patch * inner).sum(); co = (patch * outer).sum()
                    sc = c0 - 0.5 * (ci + co)
                    if best is None or c0 > best[0]:
                        best = (c0, ci, co, sc)
        out.append((vol, pno, side, k, y, s) + tuple(best))
    return out

if __name__ == '__main__':
    allp = pickle.load(open(os.path.join(WORK, 'ringpeaks.pkl'), 'rb'))
    sel = [p for p in allp if p[-1] >= 0.78]
    byp = {}
    for vol, pno, side, k, y, s in sel:
        byp.setdefault((vol, pno), []).append((side, k, y, s))
    jobs = [(v, p, it) for (v, p), it in byp.items()]
    res = []
    with ProcessPoolExecutor(10) as ex:
        for o in ex.map(page_scores, jobs):
            res += o
    pickle.dump(res, open(os.path.join(WORK, 'ringpeaks2.pkl'), 'wb'))
    res.sort(key=lambda r: r[5])
    for r in res:
        if r[5] >= 0.80:
            print(r[0], r[1], r[2], r[3], r[4], 'cov %.2f  c0 %.2f in %.2f out %.2f contrast %.2f' % (r[5], r[6], r[7], r[8], r[9]))
