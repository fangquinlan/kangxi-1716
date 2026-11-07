from common import ROOT, WORK, os
from explore_layout import *
from scipy import ndimage as ndi
if __name__ == '__main__':
    vol, pno = (sys.argv[1], int(sys.argv[2]))
    doc = fitz.open(pdf_list()[vol])
    im = load_page(doc, pno)
    norm = normalize_bg(im)
    thr, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (norm < thr).astype(np.uint8)
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    dt = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
    n, lab, st, cen = cv2.connectedComponentsWithStats(closed, 8)
    mx = ndi.maximum(dt, lab, index=np.arange(n))
    mx = np.asarray(mx)
    areas = st[:, 4]
    sel = areas > 30
    hist = np.histogram(mx[1:][sel[1:]], bins=np.arange(0, 12, 0.5))
    for b, c in zip(hist[1], hist[0]):
        print(f'{b:4.1f} {c}')
    vis = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    colors = np.zeros((n, 3), np.uint8)
    for i in range(1, n):
        m = mx[i]
        colors[i] = (0, 0, 255) if m >= 5 else (0, 160, 255) if m >= 4.5 else (0, 200, 0) if m >= 4 else (200, 120, 0) if m >= 3.5 else (128, 128, 128)
    cl = colors[lab]
    msk = lab > 0
    vis[msk] = cl[msk]
    h, w = ink.shape
    cv2.imwrite(os.path.join(WORK, f'notes/comp_{vol}_{pno}.jpg'), cv2.resize(vis, (w // 2, h // 2), interpolation=cv2.INTER_AREA))
