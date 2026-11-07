"""Draw candidate boxes (with index) on a page from a candidate dir."""
from common import *
import pickle, sys
vol, pno, cdir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rec = pickle.load(open(os.path.join(WORK, cdir, vol, f'{pno:04d}.pkl'), 'rb'))
doc = fitz.open(pdf_list()[vol])
im = load_page(doc, pno)
vis = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
for ci, c in enumerate(rec['cands']):
    x0, x1 = c['x0'], c['x1']; y0 = c['y']; y1 = c['y'] + c['h']
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 4)
    cv2.putText(vis, str(ci), (x0 - 60, y0 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
for r in rec.get('rings', []):
    cv2.circle(vis, (int(r['x']), int(r['yc'])), int(r['r']), (255, 0, 0), 4)
side = sys.argv[4] if len(sys.argv) > 4 else 'all'
if side == 'L':
    vis = vis[:, :vis.shape[1] // 2 + 100]
elif side == 'R':
    vis = vis[:, vis.shape[1] // 2 - 100:]
cv2.imwrite(os.path.join(WORK, 'notes', f'ctx2_{vol}_{pno}.jpg'), cv2.resize(vis, (vis.shape[1] // 2, vis.shape[0] // 2), interpolation=cv2.INTER_AREA))
