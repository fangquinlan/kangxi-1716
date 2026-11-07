"""Regression set: reprocess the pages of the user-listed characters with the current code (into cands_test)
and render the candidate at the same position as the aligned one."""
import os, sys, pickle
os.environ.setdefault('KX_CANDS', 'cands_test')
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pipeline
from align import WORK
from render import clean_glyph
from sheet import make_sheet
LISTS = {'02': '嗸㗙夰𠯚𡕽嗿𡣐嗅𠲺𡁊𡣊㜎', '01': '㔑𠆤𠘶𠩞价𠐻𠓆决儚勱㒵𠨰𠕝㐡劈𠟈𠑺勗勳𠨓'}
out = sys.argv[1] if len(sys.argv) > 1 else 'user_defects_v2.png'
imgs, labels = [], []
for vol, chars in LISTS.items():
    A = pickle.load(open(os.path.join(WORK, f'align_{vol}.pkl'), 'rb'))
    for ch in chars:
        rs = [r for r in A['res'] if r['ch'] == ch]
        r = max(rs, key=lambda r: r['sim'])
        p = os.path.join(pipeline.CAND_DIR, vol, f"{r['pno']:04d}.pkl")
        if not os.path.exists(p):
            pipeline.process_page((vol, r['pno']))
        rec = pickle.load(open(p, 'rb'))
        c = min(rec['cands'], key=lambda c: (c['face'] != r['face']) * 1e6 + (c['col'] != r['col']) * 1e5 + abs(c['yc'] - r['yc']))
        imgs.append(clean_glyph(c, rec['thr']))
        labels.append(f"{vol} p{r['pno']}")
make_sheet(imgs, labels, S=170, cols=12, path=os.path.join(WORK, 'notes', out))
