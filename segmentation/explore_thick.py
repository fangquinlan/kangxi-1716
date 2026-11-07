from common import ROOT, WORK, os
from explore_layout import *
if __name__ == '__main__':
    vol, pno = (sys.argv[1], int(sys.argv[2]))
    doc = fitz.open(pdf_list()[vol])
    im = load_page(doc, pno)
    norm = normalize_bg(im)
    thr, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (norm < thr).astype(np.uint8)
    t = time.time()
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    dt = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
    print('dt time', round(time.time() - t, 3))
    for th in [3, 3.5, 4, 4.5, 5, 5.5, 6]:
        print(th, int((dt >= th).sum()))
    h, w = ink.shape
    vis = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    for th, col in [(3.5, (0, 200, 0)), (4.5, (0, 0, 255)), (5.5, (255, 0, 0))]:
        m = dt >= th
        vis[m] = col
    cv2.imwrite(os.path.join(WORK, f'notes/thick_{vol}_{pno}.jpg'), cv2.resize(vis, (w // 2, h // 2), interpolation=cv2.INTER_AREA))
    cv2.imwrite(os.path.join(WORK, f'notes/thick_{vol}_{pno}_crop.png'), vis[700:1500, 2150:2900])
