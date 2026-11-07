"""Run alignment for all volumes and print a summary."""
from align import *

if __name__ == '__main__':
    vols = sys.argv[1].split(',') if len(sys.argv) > 1 else sorted(VOL_RANGES)
    rows = []
    for vol in vols:
        out = run(vol)
        res = out['res']
        sims = np.array([r['sim'] for r in res if r['sim'] is not None])
        low = int((sims < 0.8).sum())
        vlow = int((sims < 0.6).sum())
        unm = sum(1 for r in res if r['match'] < 0)
        rows.append((vol, len(out['exp']), len(sims), len(out['missing']), low, vlow, unm, float(sims.mean())))
        pickle.dump({k: v for k, v in out.items() if k not in ('cands', 'sim')},
                    open(os.path.join(WORK, os.environ.get('KX_ALIGN_PREFIX', 'align_') + f'{vol}.pkl'), 'wb'))
    print('vol expected matched missing low<0.8 low<0.6 unmatched_cands mean_sim')
    for r in rows:
        print('%s %6d %6d %5d %5d %5d %5d %.4f' % r)
    tot = np.array([r[1:7] for r in rows]).sum(0)
    print('TOTAL', tot)
