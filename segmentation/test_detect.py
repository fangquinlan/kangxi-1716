from common import ROOT, WORK, os
from detect import *
import sys, time

def page_candidates(norm, thr):
    ink_full = (norm < thr).astype(np.uint8)
    faces = find_faces(ink_full)
    out = []
    for fi, f in enumerate(faces):
        comb_fit(f, ink_full)
        P = f['pitch']
        for k in range(8):
            strip, y0, xl = extract_strip(norm, f, k)
            ink, dt, (top, bot) = prepare_strip(strip, thr)
            cands, thick = detect_strip(ink, dt, P)
            for c in cands:
                crop, bb = glyph_mask(ink, thick, c, P)
                if crop is None:
                    continue
                c.update(face=fi, side=f['side'], k=k, bbox=bb, crop=crop, y0=y0, xl=xl, P=P)
                out.append(c)
    return (faces, out)
if __name__ == '__main__':
    vol = sys.argv[1]
    doc = fitz.open(pdf_list()[vol])
    for pno in [int(x) for x in sys.argv[2].split(',')]:
        t = time.time()
        im = load_page(doc, pno)
        norm = normalize_bg(im)
        thr, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        faces, cands = page_candidates(norm, thr)
        vis = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
        crops = []
        for c in cands:
            x0, yy0, x1, yy1 = c['bbox']
            X0 = c['xl'][yy0] + x0
            X1 = c['xl'][yy1] + x1
            col = (0, 0, 255) if c['thick_area'] >= 300 else (0, 180, 255)
            cv2.rectangle(vis, (int(X0), int(yy0 + c['y0'])), (int(X1), int(yy1 + c['y0'])), col, 4)
            cv2.putText(vis, str(c['thick_area']), (int(X0), int(yy0 + c['y0']) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
            g = (1 - c['crop']) * 255
            crops.append(np.pad(g.astype(np.uint8), 10, constant_values=255))
        print(pno, 'time', round(time.time() - t, 3), 'n', len(crops))
        h, w = norm.shape
        cv2.imwrite(os.path.join(WORK, f'notes/det_{vol}_{pno}.jpg'), cv2.resize(vis, (w // 3, h // 3), interpolation=cv2.INTER_AREA))
        if crops:
            S = 220
            cols = 12
            rows = (len(crops) + cols - 1) // cols
            sheet = np.full((rows * S, cols * S), 180, np.uint8)
            for i, g in enumerate(crops):
                gh, gw = g.shape
                sc = min(1.0, (S - 4) / max(gh, gw))
                g2 = cv2.resize(g, (int(gw * sc), int(gh * sc)), interpolation=cv2.INTER_AREA) if sc < 1 else g
                r, cc = divmod(i, cols)
                sheet[r * S:r * S + g2.shape[0], cc * S:cc * S + g2.shape[1]] = g2
            cv2.imwrite(os.path.join(WORK, f'notes/crops_{vol}_{pno}.png'), sheet)
