"""Stage 2: align candidates of a volume to the expected headword sequence from the JSON.

The expected sequence is augmented with optional pseudo-entries for section titles
('X部' where the radical changes, and '康熙字典 + 卷名' at each juan start) so that title
glyphs are absorbed by the alignment instead of being matched to real headwords.
"""
from common import *
from feats import *
import pickle, sys, time
import numba

JSON = os.environ.get('KX_HEADWORDS', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sources', 'headwords.json')))
VOL_RANGES = {
    '01': (312, 793), '02': (794, 1309), '03': (1310, 1795), '04': (1796, 2333), '05': (2334, 2883),
    '06': (2884, 3465), '07': (3466, 4191), '08': (4192, 4871), '09': (4872, 5423), '10': (5424, 6203),
    '11': (6204, 6859), '12': (6860, 7413), '13': (7414, 7738),
}
_JSON = None


def jdata():
    global _JSON
    if _JSON is None:
        _JSON = json.load(open(JSON, encoding='utf-8'))
    return _JSON


def expected(vol):
    """Real entries only: list of (site_page, seq, char)."""
    d = jdata()
    a, b = VOL_RANGES[vol]
    out = []
    for p in range(a, b + 1):
        s = d['pages'].get(str(p), '')
        for i, ch in enumerate(s):
            out.append((p, i + 1, ch))
    return out


def expected_with_titles(vol):
    """Entries (page, seq, char, kind) with kind 'real' or 'title'."""
    d = jdata()
    chars, pj = d['chars'], d['page_juan']
    real = expected(vol)
    out = []
    prev_rad = None
    prev_juan = None
    for j, (p, s, ch) in enumerate(real):
        juan = pj.get(str(p), '')
        if juan != prev_juan:
            name = juan.replace(' ', '')
            if name.startswith('補遺') or name.startswith('備考'):
                name = name[2:] + name[:2]
            for c in '康熙字典' + name:
                out.append((p, 0, c, 'title'))
            prev_juan = juan
        rad = chars.get(ch, {}).get('radical')
        nxt = real[j + 1][2] if j + 1 < len(real) else None
        nrad = chars.get(nxt, {}).get('radical') if nxt else None
        if rad and rad != prev_rad and not (nrad is not None and nrad == prev_rad and nrad != rad):
            out.append((p, 0, rad, 'title'))
            out.append((p, 0, '部', 'title'))
            prev_rad = rad
        elif prev_rad is None:
            prev_rad = rad
        out.append((p, s, ch, 'real'))
    return out


def load_templates():
    z = np.load(os.path.join(WORK, 'templates.npz'))
    chars = list(z['chars'])
    idx = {c: i for i, c in enumerate(chars)}
    return idx, z['feats'], z['aspect']


_EXTRA = {}


def template_feature(ch, tidx, tfe):
    if ch in tidx:
        return tfe[tidx[ch]].astype(np.float32)
    if ch not in _EXTRA:
        from templates import render, FONT, SIZE
        from PIL import ImageFont
        b = render(ch, ImageFont.truetype(FONT, SIZE))
        _EXTRA[ch] = feature(tight(b)) if b.sum() else np.zeros(DIM, np.float32)
    return _EXTRA[ch]


def load_cands(vol):
    d = os.path.join(WORK, os.environ.get('KX_CANDS', 'cands'), vol)
    recs = []
    for f in sorted(os.listdir(d)):
        if f.endswith('.pkl'):
            with open(os.path.join(d, f), 'rb') as fh:
                recs.append(pickle.load(fh))
    return recs


def unpack(c):
    h, w = c['crop_shape']
    return np.unpackbits(c['crop'], axis=1)[:, :w]


@numba.njit(cache=True)
def dp_align(sim, tau, skip_d, skip_e):
    """skip_e: per-entry cost of leaving an expected entry unmatched."""
    n, m = sim.shape
    D = np.zeros((n + 1, m + 1), np.float32)
    T = np.zeros((n + 1, m + 1), np.uint8)
    for i in range(1, n + 1):
        D[i, 0] = D[i - 1, 0] - skip_d
        T[i, 0] = 1
    for j in range(1, m + 1):
        D[0, j] = D[0, j - 1] - skip_e[j - 1]
        T[0, j] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = D[i - 1, j - 1] + sim[i - 1, j - 1] - tau
            t = 0
            v = D[i - 1, j] - skip_d
            if v > best:
                best = v
                t = 1
            v = D[i, j - 1] - skip_e[j - 1]
            if v > best:
                best = v
                t = 2
            D[i, j] = best
            T[i, j] = t
    match = np.full(n, -1, np.int64)
    i, j = n, m
    while i > 0 and j > 0:
        t = T[i, j]
        if t == 0:
            match[i - 1] = j - 1
            i -= 1
            j -= 1
        elif t == 1:
            i -= 1
        else:
            j -= 1
    return match


BRANCHES = '子丑寅卯辰巳午未申酉戌亥'


def header_columns(cands, F, tf):
    """Candidates in header columns (康熙字典 / X集...): may only match real entries at very high similarity."""
    s_kang, s_xi, s_ji = F @ tf('康'), F @ tf('熙'), F @ tf('集')
    s_branch = np.max(np.stack([F @ tf(c) for c in BRANCHES]), 0)
    title = np.zeros(len(cands), bool)
    cols = {}
    for i, c in enumerate(cands):
        cols.setdefault((c['pno'], c['face'], c['col']), []).append(i)
    for key, idx in cols.items():
        idx = sorted(idx, key=lambda i: cands[i]['yc'])
        for a, b in zip(idx, idx[1:]):
            if (s_kang[a] >= 0.75 and s_xi[b] >= 0.75) or (s_branch[a] >= 0.75 and s_ji[b] >= 0.75):
                title[idx] = True
                break
    return title


def build(vol):
    recs = load_cands(vol)
    cands = []
    for r in recs:
        for ci, c in enumerate(r['cands']):
            cands.append(dict(pno=r['pno'], ci=ci, face=c['face'], col=c['col'], yc=c['yc'], w=c['w'], h=c['h'],
                              P=c['P'], thick_area=c['thick_area'], bin=unpack(c)))
    F = np.stack([feature(c['bin']) for c in cands]).astype(np.float32)
    return recs, cands, F


def orphan_pass(match, sim, is_real, hdr, real_cols, win=4, min_sim=0.75, margin=0.03):
    """Assign still-missing real entries to unmatched candidates close by in reading order when the entry is
    (nearly) the candidate's best match among nearby entries. Handles local order differences between the
    site's page lists and the print."""
    n = len(match)
    real_set = set(int(j) for j in real_cols)
    matched_entries = set(int(j) for j in match if j >= 0)
    ent_pos = {int(j): i for i, j in enumerate(match) if j >= 0}
    added = 0
    for j in real_cols:
        j = int(j)
        if j in matched_entries:
            continue
        prev_i = next((ent_pos[k] for k in range(j - 1, -1, -1) if k in ent_pos), 0)
        next_i = next((ent_pos[k] for k in range(j + 1, sim.shape[1]) if k in ent_pos), n - 1)
        lo, hi = max(0, prev_i - win), min(n, next_i + win + 1)
        if hi - lo > 60:
            continue
        best = None
        for i in range(lo, hi):
            if match[i] >= 0 or hdr[i]:
                continue
            s = sim[i, j]
            if s < min_sim:
                continue
            ja, jb = max(0, j - 40), min(sim.shape[1], j + 41)
            local = max(sim[i, k] for k in range(ja, jb) if k in real_set and k not in matched_entries or k == j)
            if s < local - margin:
                continue
            if best is None or s > best[0]:
                best = (s, i)
        if best:
            match[best[1]] = j
            matched_entries.add(j)
            ent_pos[j] = best[1]
            added += 1
    return added


def run(vol, tau=0.45, skip_d=0.0, skip_real=float(os.environ.get('KX_SKIP', '0.10'))):
    t = time.time()
    recs, cands, F = build(vol)
    ent = expected_with_titles(vol)
    tidx, tfe, tasp = load_templates()
    tf = lambda ch: template_feature(ch, tidx, tfe)
    E = np.stack([tf(ch) for (_, _, ch, _) in ent]).astype(np.float32)
    sim = F @ E.T
    is_real = np.array([k == 'real' for (_, _, _, k) in ent])
    hdr = header_columns(cands, F, tf)
    simdp = sim.copy()
    rows = np.nonzero(hdr)[0]
    real_cols = np.nonzero(is_real)[0]
    if len(rows):
        sub = simdp[np.ix_(rows, real_cols)]
        simdp[np.ix_(rows, real_cols)] = np.where(sub >= 0.93, sub, -1.0)
    skip_e = np.where(is_real, skip_real, 0.0).astype(np.float32)
    t1 = time.time()
    match = dp_align(simdp, tau, skip_d, skip_e)
    orph = orphan_pass(match, simdp, is_real, hdr, real_cols)
    t2 = time.time()
    exp = [ent[j][:3] for j in real_cols]
    ent2real = {int(j): k for k, j in enumerate(real_cols)}
    simr = sim[:, real_cols]
    best = simr.argmax(1)
    res = []
    for i, c in enumerate(cands):
        j = int(match[i])
        bj = int(best[i])
        r = dict(pno=c['pno'], ci=c['ci'], face=c['face'], col=c['col'], yc=c['yc'], title=bool(hdr[i]),
                 match=-1, ch=None, page=None, seq=None, sim=None,
                 best=bj, best_ch=exp[bj][2], best_sim=float(simr[i, bj]), best_page=exp[bj][0])
        if j >= 0 and is_real[j]:
            k = ent2real[j]
            r.update(match=k, ch=exp[k][2], page=exp[k][0], seq=exp[k][1], sim=float(sim[i, j]))
        elif j >= 0:
            r.update(title=True, title_ch=ent[j][2], title_sim=float(sim[i, j]), title_page=ent[j][0])
        res.append(r)
    matched = set(r['match'] for r in res if r['match'] >= 0)
    missing = [k for k in range(len(exp)) if k not in matched]
    print(vol, 'cands', len(cands), 'expected', len(exp), 'entries', len(ent), 'orphans', orph, f'{t1 - t:.1f}s dp {t2 - t1:.1f}s')
    return dict(vol=vol, cands=cands, exp=exp, res=res, missing=missing, sim=simr)


if __name__ == '__main__':
    vol = sys.argv[1]
    out = run(vol)
    res = out['res']
    sims = np.array([r['sim'] for r in res if r['sim'] is not None])
    print('matched', len(sims), 'missing', len(out['missing']), 'unmatched cands', sum(r['match'] < 0 for r in res),
          'titles', sum(1 for r in res if r.get('title_ch')))
    print('sim percentiles', np.percentile(sims, [1, 5, 10, 25, 50, 75]).round(3))
    low = sum(1 for r in res if r['sim'] is not None and r['sim'] < 0.8)
    print('low(<0.8)', low, 'mean sim', round(float(sims.mean()), 4))
    pickle.dump({k: v for k, v in out.items() if k not in ('cands', 'sim')},
                open(os.path.join(WORK, os.environ.get('KX_ALIGN', f'align_{vol}.pkl')), 'wb'))
