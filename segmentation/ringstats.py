from pipeline import *
import detect as D
import random
from concurrent.futures import ProcessPoolExecutor
def page_peaks(a):
    vol, pno = a
    doc = fitz.open(pdf_list()[vol])
    im = load_page(doc, pno)
    norm = normalize_bg(im)
    thr, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink_full = (norm < thr).astype(np.uint8)
    faces = find_faces(ink_full)
    out = []
    for f in faces:
        comb_fit(f, ink_full)
        P = f['pitch']
        for k in range(8):
            strip, y0, xl = extract_strip(norm, f, k)
            ink, dt, _ = prepare_strip(strip, thr)
            H, W = ink.shape
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
            for y in range(1, len(best) - 1):
                if best[y] > 0.7 and best[y] == best[max(0, y - 15):y + 16].max():
                    out.append((vol, pno, f['side'], k, y * 2 + y0, float(best[y])))
    return out
if __name__ == '__main__':
    random.seed(0)
    jobs = []
    for vol in ['01', '03', '05', '07', '09', '11', '13']:
        n = fitz.open(pdf_list()[vol]).page_count
        jobs += [(vol, p) for p in random.sample(range(n), 18)]
    allp = []
    with ProcessPoolExecutor(10) as ex:
        for o in ex.map(page_peaks, jobs):
            allp += o
    sc = np.array([p[-1] for p in allp])
    h, e = np.histogram(sc, bins=np.arange(0.70, 1.0001, 0.02))
    for a, b in zip(e, h): print(f'{a:.2f} {b}')
    pickle.dump(allp, open(os.path.join(WORK, 'ringpeaks.pkl'), 'wb'))
