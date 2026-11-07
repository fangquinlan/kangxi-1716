from geom import *
import pickle
from concurrent.futures import ProcessPoolExecutor
def chk(a):
    vol, pno = a
    rec = pickle.load(open(os.path.join(WORK, 'cands', vol, f'{pno:04d}.pkl'), 'rb'))
    doc = fitz.open(pdf_list()[vol])
    im = load_page(doc, pno)
    norm = normalize_bg(im)
    thr, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (norm < thr).astype(np.uint8)
    faces = find_faces(ink)
    out = []
    old = {f['side']: f for f in rec['faces']}
    for f in faces:
        comb_fit(f, ink)
        if f['side'] in old:
            d = np.abs(np.array(f['rules_u']) - np.array(old[f['side']]['rules_u'])).max()
            out.append((vol, pno, f['side'], float(d), f['pitch'], f['offset'], f['comb_score'], old[f['side']]['comb_score']))
        else:
            out.append((vol, pno, f['side'], -1.0, f['pitch'], f['offset'], f['comb_score'], 0))
    return out
if __name__ == '__main__':
    jobs = []
    for vol in sorted(os.listdir(os.path.join(WORK, 'cands'))):
        for f in sorted(os.listdir(os.path.join(WORK, 'cands', vol))):
            jobs.append((vol, int(f[:4])))
    res = []
    with ProcessPoolExecutor(10) as ex:
        for o in ex.map(chk, jobs, chunksize=4):
            res += o
    pickle.dump(res, open(os.path.join(WORK, 'combcheck.pkl'), 'wb'))
    d = np.array([r[3] for r in res])
    print('faces', len(res), 'new faces', (d < 0).sum(), 'diff>10', (d > 10).sum(), 'diff>30', (d > 30).sum())
    for r in sorted(res, key=lambda r: -r[3])[:60]:
        print(r)
