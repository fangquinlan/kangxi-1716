from align import *
from review import *
vol = sys.argv[1]
A = pickle.load(open(os.path.join(WORK, os.environ.get('KX_ALIGN', f'align_{vol}.pkl')), 'rb'))
recs = {r['pno']: r for r in load_cands(vol)}
items = []
sel = [r for r in A['res'] if r['match'] < 0]
sel.sort(key=lambda r: -r['best_sim'])
start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
for r in sel[start:start + 120]:
    c = recs[r['pno']]['cands'][r['ci']]
    g = ((1 - unpack(c)) * 255).astype(np.uint8)
    items.append((g, r['best_ch'], f"p{r['pno']} {r['best_sim']:.2f} {r['best_page']}"))
print(len(sel))
pair_sheet(items, os.path.join(WORK, 'notes', f'unm_{vol}_{start}.png'), S=90, cols=10)
