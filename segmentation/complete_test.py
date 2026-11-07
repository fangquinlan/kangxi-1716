"""Evaluate font-guided completion on random pages: report choices, show changed glyphs before/after."""
import os, sys, pickle, random
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from complete import *
from render import clean_glyph
from concurrent.futures import ProcessPoolExecutor

def work(a):
    vol, pno, items, scale = a
    rec = pickle.load(open(os.path.join(WORK, 'cands', vol, f'{pno:04d}.pkl'), 'rb'))
    doc = fitz.open(pdf_list()[vol])
    S = Strips(doc, rec)
    out = []
    for ci, ch in items:
        c = rec['cands'][ci]
        st = S.get(c['side'], c['k'])
        nc, info = complete(c, ch, st, scale)
        tile = None
        if info['choice'] != 'orig':
            a_ = clean_glyph(c, rec['thr'])
            b_ = clean_glyph(nc, rec['thr'])
            tile = (a_, b_)
        out.append((vol, pno, ch, info, tile))
    return out

if __name__ == '__main__':
    random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    npages = int(sys.argv[1])
    jobs = []
    for vol in ['01', '03', '05', '07', '09', '10', '13']:
        A = pickle.load(open(os.path.join(WORK, f'align_{vol}.pkl'), 'rb'))
        scale = pickle.load(open(os.path.join(WORK, f'repair_{vol}.pkl'), 'rb'))['scale']
        bypage = {}
        for r in A['res']:
            if r['match'] >= 0:
                bypage.setdefault(r['pno'], []).append((r['ci'], r['ch']))
        for p in random.sample(sorted(bypage), npages):
            jobs.append((vol, p, bypage[p], scale))
    res = []
    with ProcessPoolExecutor(10) as ex:
        for o in ex.map(work, jobs):
            res += o
    from collections import Counter
    print('glyphs', len(res), Counter(r[3]['choice'] for r in res))
    ch_ = [r for r in res if r[4] is not None]
    gains = sorted(ch_, key=lambda r: -(r[3]['s'] - r[3]['s0']))
    from sheet import make_sheet
    imgs, labels = [], []
    for vol, pno, ch, info, (a_, b_) in gains[:60]:
        imgs += [a_, b_]
        labels += [f"{vol}p{pno} {info['s0']:.2f}", f"{info['choice']} {info['s']:.2f}"]
    make_sheet(imgs, labels, S=150, cols=12, path=os.path.join(WORK, 'notes', 'complete_test.png'))
    imgs, labels = [], []
    for vol, pno, ch, info, (a_, b_) in gains[-40:]:
        imgs += [a_, b_]
        labels += [f"{vol}p{pno} {info['s0']:.2f}", f"{info['choice']} {info['s']:.2f}"]
    make_sheet(imgs, labels, S=150, cols=12, path=os.path.join(WORK, 'notes', 'complete_test_small.png'))
