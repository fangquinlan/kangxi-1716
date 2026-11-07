"""Draw candidate boxes with alignment labels on a page region."""
from align import *
from geom import normalize_bg
from PIL import Image, ImageDraw, ImageFont
vol = sys.argv[1]; pno = int(sys.argv[2])
A = pickle.load(open(os.path.join(WORK, os.environ.get('KX_ALIGN', f'align_{vol}.pkl')), 'rb'))
rec = pickle.load(open(os.path.join(WORK, os.environ.get('KX_CANDS', 'cands'), vol, f'{pno:04d}.pkl'), 'rb'))
doc = fitz.open(pdf_list()[vol])
im = load_page(doc, pno)
vis = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_GRAY2RGB))
d = ImageDraw.Draw(vis)
f = ImageFont.truetype(os.path.join(ROOT, 'KangXiDict_Full-Regular.ttf'), 60)
fs = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 30)
res = {(r['pno'], r['ci']): r for r in A['res']}
for ci, c in enumerate(rec['cands']):
    r = res.get((pno, ci))
    x0, x1 = c['x0'], c['x1']; y0 = c['y']; y1 = c['y'] + c['h']
    col = (255, 0, 0) if r and r['match'] >= 0 else (0, 120, 255)
    d.rectangle([x0, y0, x1, y1], outline=col, width=5)
    if r and r['match'] >= 0:
        d.text((x0 - 70, y0), r['ch'], fill=(255, 0, 0), font=f)
        d.text((x0 - 70, y0 + 65), f"{r['sim']:.2f}", fill=(255, 0, 0), font=fs)
    elif r:
        d.text((x0 - 70, y0), r['best_ch'], fill=(0, 120, 255), font=f)
        d.text((x0 - 70, y0 + 65), f"{r['best_sim']:.2f}", fill=(0, 120, 255), font=fs)
# expected chars for pages matched on this pdf page
pages = sorted(set(r['page'] for r in A['res'] if r['pno'] == pno and r['page']))
exp = A['exp']
txt = ' | '.join(f"{p}:" + ''.join(ch for (pp, s, ch) in exp if pp == p) for p in pages)
d.text((20, 20), txt, fill=(0, 0, 0), font=ImageFont.truetype(os.path.join(ROOT, 'KangXiDict_Full-Regular.ttf'), 50))
vis = vis.resize((vis.width // 2, vis.height // 2))
vis.save(os.path.join(WORK, 'notes', f'ctx_{vol}_{pno}.jpg'), quality=85)
print(pages)
