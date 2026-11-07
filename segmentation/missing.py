"""Show where missing expected entries should be (between matched neighbours)."""
from align import *
from review import render_char
from sheet import make_sheet
vol = sys.argv[1]
A = pickle.load(open(os.path.join(WORK, os.environ.get('KX_ALIGN', f'align_{vol}.pkl')), 'rb'))
exp = A['exp']
m2c = {r['match']: r for r in A['res'] if r['match'] >= 0}
recs = {r['pno']: r for r in load_cands(vol)}
doc = fitz.open(pdf_list()[vol])
items, labels = [], []
for j in A['missing'][:int(sys.argv[2]) if len(sys.argv) > 2 else 40]:
    prev = next((m2c[i] for i in range(j - 1, -1, -1) if i in m2c), None)
    nxt = next((m2c[i] for i in range(j + 1, len(exp)) if i in m2c), None)
    print(j, exp[j], 'prev', prev and (prev['pno'], prev['face'], prev['col'], round(prev['yc']), prev['ch']),
          'next', nxt and (nxt['pno'], nxt['face'], nxt['col'], round(nxt['yc']), nxt['ch']))
    # show the region after prev candidate (same column, below) as a crop of the page
    ref = prev or nxt
    rec = recs[ref['pno']]
    c = rec['cands'][ref['ci']]
    im = load_page(doc, ref['pno'])
    x0 = max(0, c['x0'] - 260); x1 = min(im.shape[1], c['x1'] + 260)
    y0 = max(0, int(c['y']) - 200); y1 = min(im.shape[0], int(c['y']) + 900)
    crop = im[y0:y1, x0:x1]
    items.append(cv2.resize(crop, (crop.shape[1] // 2, crop.shape[0] // 2)))
    labels.append(f"{j} p{ref['pno']}")
make_sheet(items, labels, S=560, cols=8, path=os.path.join(WORK, 'notes', f'missing_{vol}.png'))
