"""Render this pipeline's version of the user-listed characters (regression set)."""
from align import *
from render import clean_glyph
from sheet import make_sheet
LISTS = {'02': '嗸㗙夰𠯚𡕽𧶜嗿𡣐嗅𠲺𡁊𡣊㜎', '01': '㔑𥮏𠆤𠘶𠩞价𠐻𠓆决儚勱㒵𠨰𠕝㐡劈𠟈𠑺勗勳𠨓'}
out = sys.argv[1] if len(sys.argv) > 1 else 'user_defects_new.png'
imgs, labels = [], []
for vol, chars in LISTS.items():
    A = pickle.load(open(os.path.join(WORK, f'align_{vol}.pkl'), 'rb'))
    recs = {}
    for ch in chars:
        rs = [r for r in A['res'] if r['ch'] == ch]
        if not rs:
            print('not matched', vol, ch)
            continue
        r = max(rs, key=lambda r: r['sim'])
        if r['pno'] not in recs:
            recs[r['pno']] = pickle.load(open(os.path.join(WORK, os.environ.get('KX_CANDS', 'cands'), vol, f"{r['pno']:04d}.pkl"), 'rb'))
        rec = recs[r['pno']]
        img = clean_glyph(rec['cands'][r['ci']], rec['thr'])
        imgs.append(img)
        labels.append(f"{vol} p{r['pno']} {r['sim']:.2f}")
make_sheet(imgs, labels, S=170, cols=12, path=os.path.join(WORK, 'notes', out))
print(len(imgs))
