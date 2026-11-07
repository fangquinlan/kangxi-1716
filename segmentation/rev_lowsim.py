from align import *
from review import *
vol = sys.argv[1]; lo, hi = float(sys.argv[2]), float(sys.argv[3])
A = pickle.load(open(os.path.join(WORK, os.environ.get('KX_ALIGN', f'align_{vol}.pkl')), 'rb'))
recs = {r['pno']: r for r in load_cands(vol)}
items = []
sel = [r for r in A['res'] if r['sim'] is not None and lo <= r['sim'] < hi]
print(len(sel))
for r in sel[:int(sys.argv[4]) if len(sys.argv) > 4 else 96]:
    c = recs[r['pno']]['cands'][r['ci']]
    g = ((1 - unpack(c)) * 255).astype(np.uint8)
    items.append((g, r['ch'], f"p{r['pno']} {r['ch']} {r['sim']:.2f} b:{r['best_ch']}{r['best_sim']:.2f}"))
pair_sheet(items, os.path.join(WORK, 'notes', f'low_{vol}_{lo}_{hi}.png'))
