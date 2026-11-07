from align import *
from review import *
lo, hi = float(sys.argv[1]), float(sys.argv[2])
vols = sys.argv[3].split(',')
items = []
for vol in vols:
    A = pickle.load(open(os.path.join(WORK, f'align_{vol}.pkl'), 'rb'))
    sel = [r for r in A['res'] if r['sim'] is not None and lo <= r['sim'] < hi]
    if not sel:
        continue
    recs = {r['pno']: r for r in load_cands(vol)}
    for r in sel:
        c = recs[r['pno']]['cands'][r['ci']]
        g = ((1 - unpack(c)) * 255).astype(np.uint8)
        items.append((g, r['ch'], f"{vol}p{r['pno']} {r['sim']:.2f} b{r['best_sim']:.2f}"))
print(len(items))
pair_sheet(items, os.path.join(WORK, 'notes', sys.argv[4]), S=90, cols=10)
