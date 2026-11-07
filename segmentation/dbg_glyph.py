"""Debug overlay: strip region with ink (gray), thick (red), glyph mask (blue), cluster rows."""
from pipeline import *
from sheet import make_sheet
import pipeline as PL

def debug_cands(vol, pno, chars_or_idx, A=None):
    doc = fitz.open(pdf_list()[vol])
    im = load_page(doc, pno)
    norm = normalize_bg(im)
    thr, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink_full = (norm < thr).astype(np.uint8)
    faces = find_faces(ink_full)
    for f in faces: comb_fit(f, ink_full)
    rec = pickle.load(open(os.path.join(WORK, 'cands', vol, f'{pno:04d}.pkl'), 'rb'))
    outs = []
    for ci in chars_or_idx:
        c = rec['cands'][ci]
        f = [f for f in faces if f['side'] == c['side']][0]
        strip, y0, xl = extract_strip(norm, f, c['k'])
        ink, dt, _ = prepare_strip(strip, thr)
        ink, rings = remove_rings(ink, f['pitch'])
        closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        dt = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
        cands, thick = detect_strip(ink, dt, f['pitch'])
        bx0, by0, bx1, by1 = c['strip_box']
        a, b = max(0, by0 - 150), min(strip.shape[0], by1 + 150)
        vis = cv2.cvtColor(strip[a:b], cv2.COLOR_GRAY2BGR)
        vis[ink[a:b] > 0] = (150, 150, 150)
        vis[thick[a:b] > 0] = (0, 0, 255)
        m = np.zeros_like(ink)
        h, w = c['crop_shape']
        m[by0:by1 + 1, bx0:bx1 + 1] = np.unpackbits(c['crop'], axis=1)[:, :w]
        vis[(m[a:b] > 0) & (thick[a:b] == 0)] = (255, 120, 0)
        for cc in cands:
            cv2.line(vis, (0, cc['ty0'] - a), (6, cc['ty0'] - a), (0, 200, 0), 2)
            cv2.line(vis, (0, cc['ty1'] - a), (6, cc['ty1'] - a), (0, 120, 0), 2)
        outs.append(vis)
    return outs

if __name__ == '__main__':
    vol = sys.argv[1]
    A = pickle.load(open(os.path.join(WORK, f'align_{vol}.pkl'), 'rb'))
    targets = sys.argv[2].split(',')   # entries like pno:char
    imgs, labels = [], []
    for t in targets:
        pno, ch = t.split(':')
        pno = int(pno)
        rs = [r for r in A['res'] if r['pno'] == pno and (r['ch'] == ch or r['best_ch'] == ch)]
        for r in rs[:1]:
            v = debug_cands(vol, pno, [r['ci']])[0]
            imgs.append(cv2.cvtColor(v, cv2.COLOR_BGR2GRAY) if False else v)
            labels.append(t)
    # color sheet
    S = 420
    sheet = np.full((S + 30, S * len(imgs), 3), 200, np.uint8)
    for i, v in enumerate(imgs):
        sc = min(1.0, (S - 4) / max(v.shape[:2]))
        v2 = cv2.resize(v, (int(v.shape[1] * sc), int(v.shape[0] * sc)))
        sheet[:v2.shape[0], i * S:i * S + v2.shape[1]] = v2
        cv2.putText(sheet, labels[i], (i * S + 5, S + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.imwrite(os.path.join(WORK, 'notes', 'dbg_glyph.png'), sheet)
