"""Visualize the reading-order neighbourhood of headwords still missing after the targeted search."""
from align import *
from sheet import make_sheet
items, labels = [], []
for vol in sys.argv[1].split(','):
    A = pickle.load(open(os.path.join(WORK, f'align_{vol}.pkl'), 'rb'))
    R = pickle.load(open(os.path.join(WORK, f'repair_{vol}.pkl'), 'rb'))['found']
    ok = set(f['j'] for f in R if not f['rejected'])
    exp = A['exp']
    m2c = {r['match']: r for r in A['res'] if r['match'] >= 0}
    recs = {r['pno']: r for r in load_cands(vol)}
    doc = fitz.open(pdf_list()[vol])
    for j in A['missing']:
        if j in ok:
            continue
        prev = next((m2c[i] for i in range(j - 1, -1, -1) if i in m2c), None)
        nxt = next((m2c[i] for i in range(j + 1, len(exp)) if i in m2c), None)
        print(vol, j, exp[j], 'prev', prev and (prev['pno'], prev['face'], prev['col'], round(prev['yc']), prev['ch'], round(prev['sim'], 2)),
              'next', nxt and (nxt['pno'], nxt['face'], nxt['col'], round(nxt['yc']), nxt['ch'], round(nxt['sim'], 2)))
        ref = prev or nxt
        c = recs[ref['pno']]['cands'][ref['ci']]
        im = load_page(doc, ref['pno'])
        x0 = max(0, c['x0'] - 450); x1 = min(im.shape[1], c['x1'] + 250)
        y0 = max(0, int(c['y']) - 150); y1 = min(im.shape[0], int(c['y']) + 1000)
        crop = im[y0:y1, x0:x1]
        items.append(cv2.resize(crop, (crop.shape[1] // 2, crop.shape[0] // 2)))
        labels.append(f"{vol} {exp[j][2]} p{ref['pno']}")
make_sheet(items, labels, S=580, cols=6, path=os.path.join(WORK, 'notes', 'missing2.png'))
