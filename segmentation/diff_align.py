"""Show candidates whose assignment differs between two alignment runs."""
from align import *
from review import render_char, fit
from PIL import Image, ImageDraw, ImageFont
a_pre, b_pre, vols = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
rows = []
for vol in vols:
    A = pickle.load(open(os.path.join(WORK, f'{a_pre}{vol}.pkl'), 'rb'))
    B = pickle.load(open(os.path.join(WORK, f'{b_pre}{vol}.pkl'), 'rb'))
    recs = None
    for ra, rb in zip(A['res'], B['res']):
        if ra['ch'] != rb['ch']:
            if recs is None:
                recs = {r['pno']: r for r in load_cands(vol)}
            c = recs[ra['pno']]['cands'][ra['ci']]
            rows.append((vol, ra, rb, ((1 - unpack(c)) * 255).astype(np.uint8)))
print(len(rows))
S = 100
W = 3 * S + 20
cols = 5
H = ((len(rows) + cols - 1) // cols) * (S + 18)
sheet = Image.new('L', (cols * W, max(H, 10)), 200)
d = ImageDraw.Draw(sheet)
lf = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 12)
for i, (vol, ra, rb, g) in enumerate(rows):
    r, c = divmod(i, cols)
    x, y = c * W, r * (S + 18)
    sheet.paste(Image.fromarray(fit(g, S)), (x, y))
    if ra['ch']:
        sheet.paste(Image.fromarray(fit(render_char(ra['ch']), S)), (x + S + 4, y))
    if rb['ch']:
        sheet.paste(Image.fromarray(fit(render_char(rb['ch']), S)), (x + 2 * S + 8, y))
    sa = f"{ra['sim']:.2f}" if ra['sim'] else '-'
    sb = f"{rb['sim']:.2f}" if rb['sim'] else '-'
    d.text((x + 2, y + S + 2), f"{vol}p{ra['pno']}  A:{sa}  B:{sb}", fill=0, font=lf)
sheet.save(os.path.join(WORK, 'notes', 'diff_align.png'))
