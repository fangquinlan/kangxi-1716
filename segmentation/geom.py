"""Face geometry: find the two text frames on a spread and the 8 big columns of each."""
from common import *
from scipy.ndimage import gaussian_filter1d


def normalize_bg(im):
    h, w = im.shape
    small = cv2.resize(im, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    bg = cv2.GaussianBlur(bg, (0, 0), 3)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.divide(im, bg, scale=255)


def _comps(mask):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    return n, lab, st


def _fit_line_h(lab, idx, scale):
    """fit y = a*x + b over pixels of a horizontal component (coords scaled to full res)"""
    ys, xs = np.nonzero(lab == idx)
    xb = xs // 4
    order = np.argsort(xb, kind='stable')
    xs, ys, xb = xs[order], ys[order], xb[order]
    _, start = np.unique(xb, return_index=True)
    ends = list(start[1:]) + [len(xb)]
    cx = np.array([xs[s:e].mean() for s, e in zip(start, ends)]) * scale
    cy = np.array([np.median(ys[s:e]) for s, e in zip(start, ends)]) * scale
    a, b = np.polyfit(cx, cy, 1)
    res = np.abs(cy - (a * cx + b))
    keep = res < max(6, np.percentile(res, 80))
    a, b = np.polyfit(cx[keep], cy[keep], 1)
    return float(a), float(b)


def _fit_line_v(lab, idx, scale):
    """fit x = c*y + d over pixels of a vertical component"""
    ys, xs = np.nonzero(lab == idx)
    yb = ys // 4
    order = np.argsort(yb, kind='stable')
    xs, ys, yb = xs[order], ys[order], yb[order]
    _, start = np.unique(yb, return_index=True)
    ends = list(start[1:]) + [len(yb)]
    cy = np.array([ys[s:e].mean() for s, e in zip(start, ends)]) * scale
    cx = np.array([np.median(xs[s:e]) for s, e in zip(start, ends)]) * scale
    c, d = np.polyfit(cy, cx, 1)
    res = np.abs(cx - (c * cy + d))
    keep = res < max(6, np.percentile(res, 80))
    c, d = np.polyfit(cy[keep], cx[keep], 1)
    return float(c), float(d)


def _group_segments(st, idxs, axis, tol, s):
    """Group collinear segments. axis='h': group by y-center; 'v': by x-center. Returns list of index lists."""
    if not idxs:
        return []
    key = (lambda i: (st[i, 1] + st[i, 3] / 2) * s) if axis == 'h' else (lambda i: (st[i, 0] + st[i, 2] / 2) * s)
    idxs = sorted(idxs, key=key)
    groups = [[idxs[0]]]
    for i in idxs[1:]:
        if key(i) - np.mean([key(j) for j in groups[-1]]) <= tol:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def _fit_group_h(lab, ids, scale):
    m = np.isin(lab, ids)
    ys, xs = np.nonzero(m)
    xb = xs // 4
    order = np.argsort(xb, kind='stable')
    xs, ys, xb = xs[order], ys[order], xb[order]
    _, start = np.unique(xb, return_index=True)
    ends = list(start[1:]) + [len(xb)]
    cx = np.array([xs[a:b].mean() for a, b in zip(start, ends)]) * scale
    cy = np.array([np.median(ys[a:b]) for a, b in zip(start, ends)]) * scale
    a, b = np.polyfit(cx, cy, 1)
    res = np.abs(cy - (a * cx + b))
    keep = res < max(6, np.percentile(res, 80))
    a, b = np.polyfit(cx[keep], cy[keep], 1)
    return float(a), float(b)


def _fit_group_v(lab, ids, scale):
    m = np.isin(lab, ids)
    ys, xs = np.nonzero(m)
    yb = ys // 4
    order = np.argsort(yb, kind='stable')
    xs, ys, yb = xs[order], ys[order], yb[order]
    _, start = np.unique(yb, return_index=True)
    ends = list(start[1:]) + [len(yb)]
    cy = np.array([ys[a:b].mean() for a, b in zip(start, ends)]) * scale
    cx = np.array([np.median(xs[a:b]) for a, b in zip(start, ends)]) * scale
    c, d = np.polyfit(cy, cx, 1)
    res = np.abs(cx - (c * cy + d))
    keep = res < max(6, np.percentile(res, 80))
    c, d = np.polyfit(cy[keep], cx[keep], 1)
    return float(c), float(d)


def find_faces(ink):
    """ink: uint8 0/1 full-res mask. Returns list of face dicts (right face first)."""
    h, w = ink.shape
    s = 2
    ink2 = cv2.resize(ink * 255, (w // s, h // s), interpolation=cv2.INTER_AREA)
    ink2 = (ink2 > 64).astype(np.uint8)
    H = cv2.morphologyEx(ink2, cv2.MORPH_OPEN, np.ones((1, 80), np.uint8))
    V = cv2.morphologyEx(ink2, cv2.MORPH_OPEN, np.ones((80, 1), np.uint8))
    H = cv2.morphologyEx(H, cv2.MORPH_CLOSE, np.ones((3, 25), np.uint8))
    V = cv2.morphologyEx(V, cv2.MORPH_CLOSE, np.ones((25, 3), np.uint8))
    nh, labh, sth = _comps(H)
    nv, labv, stv = _comps(V)
    hl = [i for i in range(1, nh) if sth[i, 2] * s >= 150 and sth[i, 1] * s > 150
          and (sth[i, 1] + sth[i, 3]) * s < h - 150]
    vl = [i for i in range(1, nv) if stv[i, 3] * s >= 150]
    faces = []
    mid = w / 2
    for side in ('R', 'L'):
        cand = [i for i in hl if ((sth[i, 0] + sth[i, 2] / 2) * s > mid) == (side == 'R')]
        groups = []
        for g in _group_segments(sth, cand, 'h', 22, s):
            x0 = min(sth[i, 0] for i in g) * s
            x1 = max(sth[i, 0] + sth[i, 2] for i in g) * s
            tot = sum(sth[i, 2] for i in g) * s
            if x1 - x0 >= 900 and tot >= 700:
                y = np.mean([(sth[i, 1] + sth[i, 3] / 2) for i in g]) * s
                area = sum(sth[i, 4] for i in g)
                groups.append(dict(ids=g, x0=x0, x1=x1, tot=tot, y=y, area=area))
        tops = [g for g in groups if g['y'] < h * 0.45]
        bots = [g for g in groups if g['y'] > h * 0.6]
        best = None
        for t in tops:
            for b in bots:
                ov = min(t['x1'], b['x1']) - max(t['x0'], b['x0'])
                if ov < 800:
                    continue
                dy = b['y'] - t['y']
                if not (1800 < dy < 2700):
                    continue
                score = min(t['tot'], b['tot']) - abs(dy - 2260) * 0.5 + (t['area'] + b['area']) * 0.02
                if best is None or score > best[0]:
                    best = (score, t, b)
        if best is None:
            continue
        _, t, b = best
        at, bt = _fit_group_h(labh, t['ids'], s)
        ab, bb = _fit_group_h(labh, b['ids'], s)
        x0 = max(t['x0'], b['x0'])
        x1 = min(t['x1'], b['x1'])
        gx_target = min(t['x0'], b['x0']) if side == 'R' else max(t['x1'], b['x1'])
        # vertical line groups between the borders
        vc = []
        for i in vl:
            vx = (stv[i, 0] + stv[i, 2] / 2) * s
            if abs(vx - gx_target) < 300:
                vc.append(i)
        gbest = None
        for g in _group_segments(stv, vc, 'v', 12, s):
            vx = np.mean([(stv[i, 0] + stv[i, 2] / 2) for i in g]) * s
            ytop = at * vx + bt
            ybot = ab * vx + bb
            span = sum(min(ybot, (stv[i, 1] + stv[i, 3]) * s) - max(ytop, stv[i, 1] * s) for i in g)
            if span < 1100:
                continue
            thick = np.mean([stv[i, 2] for i in g]) * s
            d = abs(vx - gx_target)
            sc = d - 0.02 * span - 2.0 * thick
            if d < 260 and (gbest is None or sc < gbest[0]):
                gbest = (sc, g, thick)
        if gbest is None:
            continue
        cg, dg = _fit_group_v(labv, gbest[1], s)
        faces.append(dict(side=side, top=(at, bt), bot=(ab, bb), gutter=(cg, dg), gutter_w=float(gbest[2]),
                          hx=(float(x0), float(x1))))
    return faces


def gutter_x(face, y):
    c, d = face['gutter']
    return c * y + d


def comb_fit(face, ink):
    """Find rule positions as distances u from the gutter line along each row.
    Uses rule-line peaks plus the whitespace valleys between sub-columns; first rule constrained to 200..236."""
    at, bt = face['top']
    ab, bb = face['bot']
    side = face['side']
    xg_mid = gutter_x(face, 1600)
    ytop = int(at * xg_mid + bt) + 40
    ybot = int(ab * xg_mid + bb) - 40
    U = 1700
    prof = np.zeros(U, np.float64)
    ys = np.arange(ytop, ybot, 2)
    for y in ys:
        xg = gutter_x(face, y)
        if side == 'R':
            x0 = int(round(xg))
            seg = ink[y, x0:x0 + U]
        else:
            x1 = int(round(xg))
            x0 = max(0, x1 - U)
            seg = ink[y, x0:x1][::-1]
        prof[:len(seg)] += seg
    prof /= len(ys)
    hp = prof - gaussian_filter1d(prof, 6)
    sm = gaussian_filter1d(prof, 5)
    hpmax = np.array([hp[max(0, i - 2):i + 3].max() for i in range(U)])
    best = None
    for P in np.arange(186.0, 197.01, 0.25):
        for r0 in np.arange(200.0, 236.01, 1.0):
            o = r0 - P
            rules = o + P * np.arange(1, 9)
            if rules[-1] >= U - 8:
                continue
            ri = np.round(rules).astype(int)
            rule_term = hpmax[ri[:7]].sum() + 0.5 * hpmax[ri[7]]
            bnd = np.round(o + P / 2 * np.arange(1, 16)).astype(int)
            cen = np.round(o + P / 4 + P / 2 * np.arange(0, 16)).astype(int)
            valley = sm[cen].mean() - sm[bnd].mean()
            sc = rule_term + 4.0 * valley - 0.01 * abs(P - 191.3)
            if best is None or sc > best[0]:
                best = (sc, P, o, rule_term, valley)
    sc, P, o, rule_term, valley = best
    rules = []
    for k in range(1, 9):
        c = int(round(o + P * k))
        win = hp[c - 5:c + 6]
        # only snap to a real rule peak
        if win.max() > 0.08:
            rules.append(float(c - 5 + int(np.argmax(win))))
        else:
            rules.append(float(o + P * k))
    face.update(pitch=float(P), offset=float(o), rules_u=rules, comb_score=float(sc), rule_term=float(rule_term),
                valley=float(valley), ytop=ytop - 40, ybot=ybot + 40)
    return face
