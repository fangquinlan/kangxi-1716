from explore_layout import *
def comps(mask, minlen, axis):
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        L = w if axis == 'h' else h
        if L >= minlen:
            out.append((int(x), int(y), int(w), int(h), int(a)))
    return out
if __name__ == '__main__':
    vol = sys.argv[1]
    doc = fitz.open(pdf_list()[vol])
    for pno in [int(x) for x in sys.argv[2].split(',')]:
        im = load_page(doc, pno)
        norm = normalize_bg(im)
        thr, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ink = (norm < thr).astype(np.uint8)
        h, w = ink.shape
        ink2 = cv2.resize(ink * 255, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        ink2 = (ink2 > 64).astype(np.uint8)
        hk = cv2.morphologyEx(ink2, cv2.MORPH_OPEN, np.ones((1, 80), np.uint8))
        vk = cv2.morphologyEx(ink2, cv2.MORPH_OPEN, np.ones((80, 1), np.uint8))
        print(pno, 'shape', (h, w), 'thr', thr)
        print('  H:', [(x*2, y*2, ww*2, hh*2) for x, y, ww, hh, a in comps(hk, 250, 'h')])
        print('  V:', [(x*2, y*2, ww*2, hh*2) for x, y, ww, hh, a in comps(vk, 250, 'v')])
