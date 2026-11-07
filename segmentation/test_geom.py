from common import ROOT, WORK, os
from geom import *
import sys, time
vol = sys.argv[1]
doc = fitz.open(pdf_list()[vol])
for pno in [int(x) for x in sys.argv[2].split(',')]:
    t = time.time()
    im = load_page(doc, pno)
    norm = normalize_bg(im)
    thr, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (norm < thr).astype(np.uint8)
    t1 = time.time()
    faces = find_faces(ink)
    t2 = time.time()
    for f in faces:
        comb_fit(f, ink)
    t3 = time.time()
    print(pno, 'times', round(t1 - t, 3), round(t2 - t1, 3), round(t3 - t2, 3))
    vis = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    for f in faces:
        ang = (np.degrees(np.arctan(f['top'][0])), np.degrees(np.arctan(f['bot'][0])), np.degrees(np.arctan(-f['gutter'][0])))
        print('  ', f['side'], 'angles top/bot/gut', [round(a, 3) for a in ang], 'P', f['pitch'], 'o', f['offset'], 'score', round(f['comb_score'], 3), 'rules', [round(r) for r in f['rules_u']])
        sgn = 1 if f['side'] == 'R' else -1
        for k, u in enumerate([0] + f['rules_u']):
            p0 = (int(gutter_x(f, f['ytop']) + sgn * u), int(f['ytop']))
            p1 = (int(gutter_x(f, f['ybot']) + sgn * u), int(f['ybot']))
            cv2.line(vis, p0, p1, (0, 0, 255) if k in (0, 8) else (0, 160, 0), 3)
        for a, b in (f['top'], f['bot']):
            cv2.line(vis, (0, int(b)), (vis.shape[1], int(a * vis.shape[1] + b)), (255, 0, 0), 2)
    h, w = ink.shape
    cv2.imwrite(os.path.join(WORK, f'notes/geom_{vol}_{pno}.jpg'), cv2.resize(vis, (w // 3, h // 3), interpolation=cv2.INTER_AREA))
