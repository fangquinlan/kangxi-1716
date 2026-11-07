from common import *
import sys, time
from scipy.ndimage import gaussian_filter1d

def normalize_bg(im):
    h, w = im.shape
    small = cv2.resize(im, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    bg = cv2.GaussianBlur(bg, (0, 0), 3)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    norm = np.clip(im.astype(np.float32) / np.maximum(bg, 1) * 255, 0, 255).astype(np.uint8)
    return norm
if __name__ == '__main__':
    vol, pno = (sys.argv[1], int(sys.argv[2]))
    doc = fitz.open(pdf_list()[vol])
    t = time.time()
    im = load_page(doc, pno)
    norm = normalize_bg(im)
    t1 = time.time()
    thr, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (norm < thr).astype(np.uint8)
    print('otsu', thr, 'time', round(t1 - t, 3), round(time.time() - t1, 3))
    h, w = ink.shape
    ink2 = cv2.resize(ink * 255, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    ink2 = (ink2 > 64).astype(np.uint8)
    vk = cv2.morphologyEx(ink2, cv2.MORPH_OPEN, np.ones((150, 1), np.uint8))
    hk = cv2.morphologyEx(ink2, cv2.MORPH_OPEN, np.ones((1, 150), np.uint8))
    prof = vk.sum(0)
    peaks = [x for x in range(1, len(prof) - 1) if prof[x] > 100 and prof[x] >= prof[x - 1] and (prof[x] >= prof[x + 1])]
    print('vertical peaks', [(x * 2, int(prof[x])) for x in peaks])
    hprof = hk.sum(1)
    hp = [y for y in range(1, len(hprof) - 1) if hprof[y] > 150 and hprof[y] >= hprof[y - 1] and (hprof[y] >= hprof[y + 1])]
    print('horizontal peaks', [(y * 2, int(hprof[y])) for y in hp])
    vis = cv2.cvtColor(cv2.resize(norm, (w // 3, h // 3)), cv2.COLOR_GRAY2BGR)
    vv = cv2.resize(vk * 255, (w // 3, h // 3))
    hh = cv2.resize(hk * 255, (w // 3, h // 3))
    vis[vv > 0] = (0, 0, 255)
    vis[hh > 0] = (255, 0, 0)
    cv2.imwrite(os.path.join(WORK, f'notes/layout_{vol}_{pno}.jpg'), vis)
