"""Stage 2b: targeted search for expected headwords that the alignment left unmatched.

For a missing entry, the reading-order stretch between its matched neighbours is searched with the
font glyph as a template (on the cleaned column ink). The glyph is assembled from the ink components
under the best template position and accepted only if its shape matches the font.
"""
from align import *
from detect import *
from templates import render as font_render, FONT, SIZE
from PIL import ImageFont

_font = None
_ff = {}


_fb = {}


def font_bin(ch):
    global _font
    if _font is None:
        _font = ImageFont.truetype(FONT, SIZE)
    if ch not in _fb:
        b = font_render(ch, _font)
        _fb[ch] = tight(b) if b.sum() else None
    return _fb[ch]


def font_feature(ch):
    if ch not in _ff:
        fb = font_bin(ch)
        _ff[ch] = feature(fb) if fb is not None else np.zeros(DIM, np.float32)
    return _ff[ch]


def segments(prev, nxt, recs):
    """Reading-order (pno, face, col, y_from, y_to) stretches between two matched candidates."""
    lo = (prev['pno'], prev['face'], prev['col']) if prev else None
    hi = (nxt['pno'], nxt['face'], nxt['col']) if nxt else None
    segs = []
    for p in sorted(recs):
        if lo and p < lo[0] or hi and p > hi[0]:
            continue
        for fi in (0, 1):
            for col in range(8):
                key = (p, fi, col)
                if lo and key < lo or hi and key > hi:
                    continue
                y0 = prev['yc'] if lo and key == lo else -1
                y1 = nxt['yc'] if hi and key == hi else 1e9
                segs.append((p, fi, col, y0, y1))
    return segs


class StripCache:
    def __init__(self, vol):
        self.doc = fitz.open(pdf_list()[vol])
        self.page = None
        self.strips = {}

    def get(self, rec, f, k):
        p = rec['pno']
        if self.page != p:
            self.page = p
            self.norm = normalize_bg(load_page(self.doc, p))
            self.strips = {}
        key = (f['side'], k)
        if key not in self.strips:
            P = f['pitch']
            thr = rec['thr']
            strip, y0, xl = extract_strip(self.norm, f, k)
            raw = (strip < thr).astype(np.uint8)
            ink, dt, _ = prepare_strip(strip, thr)
            ink, rings = remove_rings(ink, P)
            ink, _ = remove_lines(ink, P)
            closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            dt = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
            self.strips[key] = (strip, y0, xl, ink, dt, raw & (1 - ink))
        return self.strips[key]


def assemble_at(ch, ink, y, x, th, tw):
    """Glyph = ink components lying mostly inside the template footprint at (y, x)."""
    H, W = ink.shape
    pad = 6
    by0, by1 = max(0, y - pad), min(H, y + th + pad)
    bx0, bx1 = max(0, x - pad), min(W, x + tw + pad)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink[max(0, by0 - 200):by1 + 200], 8)
    off = max(0, by0 - 200)
    m = np.zeros_like(ink)
    for i in np.unique(lab[by0 - off:by1 - off, bx0:bx1]):
        if i == 0:
            continue
        xx, yy, ww, hh, a = st[i]
        yy += off
        inter = max(0, min(xx + ww, bx1) - max(xx, bx0)) * max(0, min(yy + hh, by1) - max(yy, by0))
        if inter >= 0.6 * ww * hh:
            m[off:off + lab.shape[0]][lab == i] = 1
    ys_, xs_ = np.nonzero(m)
    if len(ys_) == 0:
        return None, -1.0
    gx0, gy0, gx1, gy1 = int(xs_.min()), int(ys_.min()), int(xs_.max()), int(ys_.max())
    crop = m[gy0:gy1 + 1, gx0:gx1 + 1]
    return (crop, (gx0, gy0, gx1, gy1)), float(feature(crop) @ font_feature(ch))


def search(ch, prev, nxt, recs, scale, sc, occupied):
    fb = font_bin(ch)
    if fb is None:
        return None
    fb = fb.astype(np.float32)
    segs = segments(prev, nxt, recs)
    if not segs or len(segs) > 24:
        return None
    best = None
    for (p, fi, col, ya, yb) in segs:
        rec = recs[p]
        face = [f for f in rec['faces'] if (0 if f['side'] == 'R' else 1) == fi]
        if not face:
            continue
        f = face[0]
        P = f['pitch']
        k = 7 - col if f['side'] == 'R' else col
        strip, y0, xl, ink, dt, forbid = sc.get(rec, f, k)
        H, W = ink.shape
        th, tw = int(round(fb.shape[0] * scale)), int(round(fb.shape[1] * scale))
        if th < 5 or tw < 5 or th >= H or tw >= W:
            continue
        tb = cv2.resize(fb, (tw, th), interpolation=cv2.INTER_AREA)
        tpl = cv2.GaussianBlur(tb, (0, 0), 2)
        img = cv2.GaussianBlur(ink.astype(np.float32), (0, 0), 2)
        res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
        res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
        # require real ink under the template (near-blank windows give unstable normalized scores)
        ii = cv2.integral(ink)
        win = (ii[th:th + res.shape[0], tw:tw + res.shape[1]] - ii[:res.shape[0], tw:tw + res.shape[1]]
               - ii[th:th + res.shape[0], :res.shape[1]] + ii[:res.shape[0], :res.shape[1]])
        res[win < 0.35 * float((tb > 0.5).sum())] = -1
        ys = np.arange(res.shape[0]) + y0 + th / 2
        valid = (ys > ya + 0.3 * P) & (ys < yb - 0.3 * P)
        xc = np.arange(res.shape[1]) + tw / 2
        res[~valid, :] = -1
        res[:, np.abs(xc - W / 2) > 0.18 * P] = -1
        for (op, ofi, ocol, oy0, oy1) in occupied:
            if (op, ofi, ocol) == (p, fi, col):
                a0 = int(max(0, oy0 - y0 - th * 0.6))
                b0 = int(min(res.shape[0], oy1 - y0 - th * 0.4))
                if b0 > a0:
                    res[a0:b0] = -1
        y, x = np.unravel_index(np.argmax(res), res.shape)
        s = float(res[y, x])
        if s < 0.35:
            continue
        g, sim = assemble_at(ch, ink, y, x, th, tw)
        if g is None:
            continue
        if best is None or sim > best[0]:
            best = (sim, s, g, p, fi, col, k, f, strip, ink, forbid, y0, xl, P, rec['thr'])
    if best is None:
        return None
    sim, s, (crop, (gx0, gy0, gx1, gy1)), p, fi, col, k, f, strip, ink, forbid, y0, xl, P, thr = best
    H, W = ink.shape
    out = dict(ch=ch, sim=sim, tm=s, pno=p, face=fi, col=col)
    if sim < 0.6:
        out['rejected'] = True
        return out
    qy0, qy1 = max(0, gy0 - 16), min(H, gy1 + 17)
    qx0, qx1 = max(0, gx0 - 16), min(W, gx1 + 17)
    fbm = forbid[qy0:qy1, qx0:qx1]
    out.update(rejected=False, side=f['side'], k=k, y=int(gy0 + y0), yc=float((gy0 + gy1) / 2 + y0),
               x0=int(xl[gy0] + gx0), x1=int(xl[gy1] + gx1), w=int(gx1 - gx0 + 1), h=int(gy1 - gy0 + 1),
               P=float(P), thick_area=0, crop=np.packbits(crop.astype(np.uint8), axis=1), crop_shape=crop.shape,
               gray=strip[qy0:qy1, qx0:qx1].copy(), gray_off=(gx0 - qx0, gy0 - qy0),
               forbid=np.packbits(fbm, axis=1), forbid_shape=fbm.shape,
               strip_box=(gx0, gy0, gx1, gy1), strip_y0=int(y0), thr=float(thr))
    return out


def run_vol(vol):
    A = pickle.load(open(os.path.join(WORK, f'align_{vol}.pkl'), 'rb'))
    exp, res = A['exp'], A['res']
    recs = {r['pno']: r for r in load_cands(vol)}
    m2r = {r['match']: r for r in res if r['match'] >= 0}
    occupied = []
    for r in res:
        if r['match'] >= 0 and r['sim'] >= 0.8:
            c = recs[r['pno']]['cands'][r['ci']]
            occupied.append((r['pno'], r['face'], r['col'], c['y'], c['y'] + c['h']))
    ratios = []
    for r in res[::5]:
        if r['match'] >= 0 and r['sim'] > 0.9:
            c = recs[r['pno']]['cands'][r['ci']]
            fb = font_bin(r['ch'])
            if fb is not None:
                ratios += [c['h'] / fb.shape[0], c['w'] / fb.shape[1]]
    scale = float(np.median(ratios))
    sc = StripCache(vol)
    found = []
    weak = {r['match']: r for r in res if r['match'] >= 0 and r['sim'] < 0.5}
    for j in sorted(set(A['missing']) | set(weak)):
        prev = next((m2r[i] for i in range(j - 1, -1, -1) if i in m2r), None)
        nxt = next((m2r[i] for i in range(j + 1, len(exp)) if i in m2r), None)
        r = search(exp[j][2], prev, nxt, recs, scale, sc, occupied)
        old = weak.get(j)
        if r is not None and not r['rejected'] and old is not None:
            # replace a weak alignment only by a clearly better glyph elsewhere
            if r['sim'] < max(0.75, old['sim'] + 0.2):
                r['rejected'] = True
            else:
                r['replaces'] = (old['pno'], old['ci'])
        tag = 'not found' if r is None else ('rejected sim %.2f' % r['sim'] if r['rejected'] else 'FOUND sim %.2f p%d' % (r['sim'], r['pno']))
        print(vol, j, exp[j], ('weak %.2f ' % old['sim']) if old else '', tag, flush=True)
        if r is not None:
            r.update(j=j, page=exp[j][0], seq=exp[j][1])
            found.append(r)
    pickle.dump(dict(scale=scale, found=found), open(os.path.join(WORK, f'repair_{vol}.pkl'), 'wb'))
    return found


if __name__ == '__main__':
    for vol in sys.argv[1].split(','):
        run_vol(vol)
