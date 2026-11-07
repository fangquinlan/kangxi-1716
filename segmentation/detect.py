"""Headword candidate detection within each big column strip."""
from geom import *
from scipy import ndimage as ndi

MARGIN = 24  # extra pixels taken on each side of a strip
DT_THICK = float(os.environ.get('KX_DT', '4.0'))
BLOB_MIN = int(os.environ.get('KX_BLOB', '20'))


def strip_bounds(face):
    P = face['pitch']
    r = face['rules_u']
    return [r[0] - P] + list(r)  # 9 boundaries, distances from gutter


def extract_strip(img, face, k, margin=MARGIN, fill=255):
    """Return strip image for column k (0 = next to gutter), its first row y0 and per-row x offsets."""
    B = strip_bounds(face)
    u0, u1 = B[k], B[k + 1]
    W = int(round(u1 - u0)) + 2 * margin
    y0 = int(max(0, face['ytop']))
    y1 = int(min(img.shape[0], face['ybot']))
    ys = np.arange(y0, y1)
    xg = gutter_x(face, ys)
    if face['side'] == 'R':
        xl = np.round(xg + u0 - margin).astype(int)
    else:
        xl = np.round(xg - u1 - margin).astype(int)
    out = np.full((len(ys), W), fill, np.uint8)
    for i, (y, x) in enumerate(zip(ys, xl)):
        a, b = max(0, x), min(img.shape[1], x + W)
        if b > a:
            out[i, a - x:b - x] = img[y, a:b]
    return out, y0, xl


def clean_lines(ink, dt, margin=MARGIN):
    """Remove frame borders and column rules (thin), keeping thick glyph strokes."""
    H, W = ink.shape
    v_long = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((300, 1), np.uint8))
    v_long = cv2.dilate(v_long, np.ones((3, 5), np.uint8))
    zone = np.zeros_like(ink)
    z = 13
    zone[:, max(0, margin - z):margin + z] = 1
    zone[:, W - margin - z:min(W, W - margin + z)] = 1
    v_short = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((15, 1), np.uint8))
    protect = cv2.dilate((dt >= 3.2).astype(np.uint8), np.ones((5, 5), np.uint8))
    v_short = v_short & zone & (1 - protect)
    v_short = cv2.dilate(v_short, np.ones((3, 3), np.uint8)) & zone & (1 - protect)
    hk = max(60, W - 2 * margin - 30)
    bridged = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((1, 9), np.uint8))
    hl = cv2.morphologyEx(bridged, cv2.MORPH_OPEN, np.ones((1, hk), np.uint8))
    hl = cv2.dilate(hl, np.ones((5, 3), np.uint8)) & ink
    # long horizontal lines only exist as frame borders at the column ends (handled by interior_rows);
    # inside the column a long horizontal run is a headword stroke
    hl[:] = 0
    out = ink.copy()
    out[(v_long > 0) | (v_short > 0) | (hl > 0)] = 0
    # also drop tiny specks left in the rule zones
    n, lab, st, _ = cv2.connectedComponentsWithStats(out & zone, 8)
    small = np.nonzero(st[:, 4] < 40)[0]
    small = small[small > 0]
    if len(small):
        out[np.isin(lab, small)] = 0
    return out, (hl > 0)


def run_width(ink, axis):
    """Length of the ink run (along axis: 1 = horizontal, 0 = vertical) through each pixel."""
    a = (ink if axis == 1 else ink.T) > 0
    H, W = a.shape
    starts = np.diff(a.astype(np.int8), axis=1, prepend=0) == 1
    ids = np.cumsum(starts, axis=1) + (np.arange(H) * (W + 1))[:, None]
    ids = np.where(a, ids, 0)
    cnt = np.bincount(ids.ravel())
    cnt[0] = 0
    out = cnt[ids].astype(np.int32)
    return out if axis == 1 else out.T


def thin_verticals(ink, P, max_w=4):
    """Long thin vertical lines (column rules, box sides): every vertical headword stroke is >= 8 px wide."""
    hw = run_width(ink, 1)
    cand = (ink > 0) & (hw <= max_w)
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 1), np.uint8))
    v = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((int(0.25 * P), 1), np.uint8))
    v = cv2.dilate(v, np.ones((1, 3), np.uint8)) & ink & (hw <= max_w + 2).astype(np.uint8)
    return v


def frame_caps(ink, dt, P, clusters, margin=MARGIN):
    """Edges of citation boxes touching/near a headword. A cap is a thin horizontal line (no thick pixels)
    lying inside one sub-column, just below a headword (top edge, sides continue downwards) or just above
    one (bottom edge, sides continue upwards). Both end continuations must be thin (no headword stroke),
    and the bands just outside and inside the line must be mostly blank.
    clusters: list of (ty0, ty1) thick-row ranges of headword candidates. Returns (removal mask, caps)."""
    H, W = ink.shape
    half = W / 2
    L = int(0.30 * P)
    hl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, L), np.uint8))
    hl = cv2.morphologyEx(hl, cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(hl, 8)
    rem = np.zeros_like(ink)
    caps = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if h > 5 or w < L:
            continue
        x1 = x + w - 1
        in_left = x >= margin - 8 and x1 <= half + 8
        in_right = x >= half - 8 and x1 <= W - margin + 8
        if not (in_left or in_right):
            continue
        comp = lab == i
        if np.percentile(dt[comp], 90) > 2.5:
            continue
        yc = y + h // 2
        for d in (1, -1):
            # must sit just below (d=1) or just above (d=-1) a headword
            if d > 0:
                near = any(t1 - 0.15 * P <= yc <= t1 + 0.35 * P for t0, t1 in clusters)
            else:
                near = any(t0 - 0.35 * P <= yc <= t0 + 0.15 * P for t0, t1 in clusters)
            if not near:
                continue
            ok = True
            for xe, sgn in ((x, -1), (x1, 1)):
                xa, xb = (xe - 16, xe + 8) if sgn < 0 else (xe - 8, xe + 16)
                ya, yb = (yc + 4, yc + 26) if d > 0 else (yc - 26, yc - 4)
                xa, xb, ya, yb = max(0, xa), min(W, xb), max(0, ya), min(H, yb)
                if xb <= xa or yb <= ya:
                    ok = False
                    break
                win = ink[ya:yb, xa:xb]
                if win.sum() < 12 or dt[ya:yb, xa:xb].max() > 2.5:
                    ok = False
                    break
            if not ok:
                continue
            xa, xb = x + 10, x1 - 10
            oa, ob = (yc - 9, yc - 3) if d > 0 else (yc + 4, yc + 10)
            ia, ib = (yc + 4, yc + 9) if d > 0 else (yc - 8, yc - 3)
            oa, ob, ia, ib = max(0, oa), min(H, ob), max(0, ia), min(H, ib)
            if ob <= oa or ib <= ia or xb <= xa:
                continue
            outer = ink[oa:ob, xa:xb].any(0).mean()
            inner = ink[ia:ib, xa:xb].any(0).mean()
            if outer > 0.45 or inner > 0.35:
                continue
            caps.append((int(x), int(y), int(w), int(h), d))
            thin = (ink > 0) & (dt <= 2.5)
            rem |= (cv2.dilate(comp.astype(np.uint8), np.ones((5, 3), np.uint8)) & thin).astype(np.uint8)
            for xe, sgn in ((x, -1), (x1, 1)):
                xa2, xb2 = (xe - 18, xe + 10) if sgn < 0 else (xe - 10, xe + 18)
                ya2, yb2 = (yc - 3, yc + 34) if d > 0 else (yc - 34, yc + 3)
                xa2, xb2, ya2, yb2 = max(0, xa2), min(W, xb2), max(0, ya2), min(H, yb2)
                rem[ya2:yb2, xa2:xb2] |= thin[ya2:yb2, xa2:xb2].astype(np.uint8)
            break
    return rem, caps


def interior_rows(raw, zone=60, cov_thr=0.85):
    """First/last interior row. Frame border rows are inked across the full strip width (they continue into
    the neighbouring columns); a headword stroke never extends past the column rules, so it cannot qualify."""
    H = raw.shape[0]
    cover = raw.mean(1)
    rows = np.nonzero(cover > cov_thr)[0]
    top, bot = 0, H
    t = rows[rows < zone]
    if len(t):
        top = int(t.max()) + 3
    b = rows[rows >= H - zone]
    if len(b):
        bot = int(b.min()) - 2
    return top, bot


def fine_clusters(thick, gap):
    rows = thick.sum(1)
    ys = np.nonzero(rows)[0]
    out = []
    if len(ys) == 0:
        return out
    start = prev = ys[0]
    for y in ys[1:]:
        if y - prev > gap:
            out.append([int(start), int(prev)])
            start = y
        prev = y
    out.append([int(start), int(prev)])
    return out


def merge_clusters(cl, P, hmax=1.02, gmax=0.36):
    cl = [list(c) for c in cl]
    while len(cl) > 1:
        best = None
        for i in range(len(cl) - 1):
            g = cl[i + 1][0] - cl[i][1]
            hgt = cl[i + 1][1] - cl[i][0]
            if g <= gmax * P and hgt <= hmax * P:
                if best is None or g < best[0]:
                    best = (g, i)
        if best is None:
            break
        i = best[1]
        cl[i] = [cl[i][0], cl[i + 1][1]]
        del cl[i + 1]
    return cl


def split_tall(cl, thick, P, hsplit=1.18):
    out = []
    for a, b in cl:
        if b - a <= hsplit * P:
            out.append([a, b])
            continue
        rows = thick[a:b + 1].sum(1)
        best = None
        y = 0
        while y < len(rows):
            if rows[y] == 0:
                s = y
                while y < len(rows) and rows[y] == 0:
                    y += 1
                mid = (s + y) / 2
                pa, pb = mid, len(rows) - mid
                if pa > 0.3 * P and pb > 0.3 * P:
                    sc = (y - s) - 0.2 * abs(pa - pb)
                    if best is None or sc > best[0]:
                        best = (sc, s, y)
            y += 1
        if best is None:
            out.append([a, b])
        else:
            _, s, e = best
            out.extend(split_tall([[a, a + s - 1], [a + e, b]], thick, P, hsplit))
    return out


_ANN = {}


def _annulus(r):
    """ring, inner and outer annulus templates (normalized) sharing one size."""
    key = round(r * 2) / 2
    if key not in _ANN:
        R = int(np.ceil(key + 7))
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        d = np.sqrt(yy ** 2 + xx ** 2)
        ring = ((d >= key - 1.0) & (d <= key + 1.0)).astype(np.float32)
        inner = ((d >= key - 6.0) & (d <= key - 4.0)).astype(np.float32)
        outer = ((d >= key + 3.5) & (d <= key + 5.5)).astype(np.float32)
        _ANN[key] = (ring / ring.sum(), inner / inner.sum(), outer / outer.sum(), R)
    return _ANN[key]


def find_rings(ink, P, cov_thr=0.78, con_thr=0.5):
    """Circled marks (thin rings ~0.75P across): annulus coverage high, blank just inside/outside.
    Returns list of (cx, cy, r, coverage) in strip coordinates."""
    H, W = ink.shape
    small = cv2.resize(ink.astype(np.float32), (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    small = (small > 0.2).astype(np.float32)
    small = cv2.dilate(small, np.ones((3, 3), np.uint8))
    h2 = H // 2
    best = np.zeros(h2, np.float32)
    bcov = np.zeros(h2, np.float32)
    bx = np.zeros(h2, np.float32)
    br = np.zeros(h2, np.float32)
    for r in np.arange(0.345 * P / 2, 0.415 * P / 2, 1.5):
        ring, inner, outer, R = _annulus(r)
        if small.shape[0] < 2 * R + 2 or small.shape[1] < 2 * R + 2:
            continue
        cxs = W // 4 - R
        lo, hi = max(0, cxs - 8), min(small.shape[1] - 2 * R, cxs + 9)
        if hi <= lo:
            continue
        region = small[:, lo:hi + 2 * R]
        cov = cv2.matchTemplate(region, ring, cv2.TM_CCORR)
        cin = cv2.matchTemplate(region, inner, cv2.TM_CCORR)
        cout = cv2.matchTemplate(region, outer, cv2.TM_CCORR)
        con = cov - 0.5 * (cin + cout)
        con[cov < cov_thr] = -1
        m = con.max(1)
        ax = con.argmax(1)
        cv_ = cov[np.arange(len(ax)), ax]
        yc = np.arange(len(m)) + R
        ok = yc < h2
        yc, m, ax, cv_ = yc[ok], m[ok], ax[ok], cv_[ok]
        upd = m > best[yc]
        best[yc[upd]] = m[upd]
        bcov[yc[upd]] = cv_[upd]
        bx[yc[upd]] = lo + ax[upd] + R
        br[yc[upd]] = round(r * 2) / 2
    rings = []
    y = 0
    while y < h2:
        if best[y] >= con_thr:
            s = y
            while y < h2 and best[y] >= con_thr * 0.9:
                y += 1
            j = s + int(np.argmax(best[s:y]))
            rings.append((float(bx[j] * 2 + 1), float(j * 2 + 1), float(br[j] * 2), float(bcov[j])))
        y += 1
    return rings


def ring_mask(shape, rings, grow=6):
    m = np.zeros(shape, np.uint8)
    for cx, cy, r, sc in rings:
        cv2.circle(m, (int(round(cx)), int(round(cy))), int(round(r + grow)), 1, -1)
    return m


def ring_removal(ink, rings):
    """Pixels to remove for circled marks: everything inside the ring and the (thin) ring line itself."""
    H, W = ink.shape
    inner = np.zeros((H, W), np.uint8)
    band = np.zeros((H, W), np.uint8)
    for cx, cy, r, sc in rings:
        c = (int(round(cx)), int(round(cy)))
        cv2.circle(inner, c, max(1, int(round(r - 4))), 1, -1)
        cv2.circle(band, c, int(round(r + 6)), 1, -1)
    band[inner > 0] = 0
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    dt = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
    thick = cv2.dilate((dt >= 3.0).astype(np.uint8), np.ones((5, 5), np.uint8))
    rem = (inner > 0) | ((band > 0) & (thick == 0))
    # components of what remains in the band that are tiny are ring debris
    return rem


def prepare_strip(strip, thr):
    ink = (strip < thr).astype(np.uint8)
    raw = ink.copy()
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    dt = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
    ink, hmask = clean_lines(ink, dt)
    top, bot = interior_rows(raw)
    ink[:top] = 0
    ink[bot:] = 0
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    dt = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
    return ink, dt, (top, bot)


def remove_rings(ink, P):
    rings = find_rings(ink, P)
    if rings:
        rem = ring_removal(ink, rings)
        ink = ink.copy()
        ink[rem] = 0
        # drop small debris left around the rings
        band = ring_mask(ink.shape, rings, grow=8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(ink & band, 8)
        small = np.nonzero(st[:, 4] < 60)[0]
        small = small[small > 0]
        if len(small):
            ink[np.isin(lab, small)] = 0
    return ink, rings


def remove_lines(ink, P):
    """Remove long thin verticals (rules, box sides) and citation-box caps next to headwords;
    thick headword strokes are protected."""
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    dt = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
    protect = cv2.dilate((dt >= 3.5).astype(np.uint8), np.ones((5, 5), np.uint8))
    v = thin_verticals(ink, P)
    rem = v & (1 - protect)
    out = ink & (1 - rem)
    return out, []


def detect_strip(ink, dt, P, margin=MARGIN, min_area=60):
    W = ink.shape[1]
    thick = (dt >= DT_THICK).astype(np.uint8)
    # ignore thick specks outside the column proper (fish tails etc.)
    thick[:, :margin + 2] = 0
    thick[:, W - margin - 2:] = 0
    # drop tiny thick blobs (junctions of small text, ink dots)
    n, lab, st, _ = cv2.connectedComponentsWithStats(thick, 8)
    small = np.nonzero(st[:, 4] < BLOB_MIN)[0]
    small = small[small > 0]
    if len(small):
        thick[np.isin(lab, small)] = 0
    cl = fine_clusters(thick, 0.12 * P)
    # drop small-text blobs: little thick area, narrow, inside one sub-column
    keep = []
    for a, b in cl:
        sub = thick[a:b + 1]
        area = int(sub.sum())
        if area < 120:
            xs = np.nonzero(sub.any(0))[0]
            spread = xs.max() - xs.min()
            cx = np.nonzero(sub)[1].mean()
            if spread < 0.3 * P or abs(cx - W / 2) > 0.15 * P:
                continue
        keep.append([a, b])
    cl = merge_clusters(keep, P, hmax=1.02, gmax=0.5)
    cl = split_tall(cl, thick, P)
    cands = []
    for a, b in cl:
        sub = thick[a:b + 1]
        area = int(sub.sum())
        if area < min_area:
            continue
        xs = np.nonzero(sub.any(0))[0]
        cx = float((np.nonzero(sub)[1]).mean())
        cands.append(dict(ty0=a, ty1=b, tx0=int(xs.min()), tx1=int(xs.max()), tcx=cx, thick_area=area))
    return cands, thick


def glyph_mask(ink, thick, c, P, dt, clip=(MARGIN - 8, MARGIN - 8)):
    """Collect the glyph's components around a thick cluster. Returns (mask crop, bbox) in strip coords.
    clip = (left, right) number of columns blanked on each side of the strip."""
    H, W = ink.shape
    a, b = c['ty0'], c['ty1']
    pad = int(0.3 * P)
    y0, y1 = max(0, a - pad), min(H, b + pad + 1)
    lim_top = max(0, int(a - 0.07 * P) - y0)
    lim_bot = min(y1 - y0 - 1, int(b + 0.2 * P) - y0)
    sub = ink[y0:y1].copy()
    sub[:, :clip[0]] = 0
    sub[:, W - clip[1]:] = 0
    n, lab, st, cen = cv2.connectedComponentsWithStats(sub, 8)
    th = (dt[y0:y1] >= DT_THICK).astype(np.uint8)
    th[:a - y0] = 0
    th[b + 1 - y0:] = 0
    th[:, :max(clip[0], MARGIN + 2)] = 0
    th[:, W - max(clip[1], MARGIN + 2):] = 0
    core_ids = np.unique(lab[th > 0])
    core_ids = core_ids[core_ids > 0]
    if len(core_ids) == 0:
        return None, None
    core = np.zeros(sub.shape, bool)
    k3 = np.ones((3, 3), np.uint8)
    inner = np.zeros(sub.shape, bool)
    inner[:, MARGIN + 4:W - MARGIN - 4] = True
    for i in core_ids:
        x, y, w, h, ar = st[i]
        comp = (lab == i)
        # components lying mostly outside the column proper (fish tails, neighbour strokes)
        if (comp & inner).sum() < 0.6 * ar:
            continue
        if y < lim_top or y + h - 1 > lim_bot:
            # touching something outside the glyph rows: split by erosion, keep thick-connected parts
            er = cv2.erode(comp.astype(np.uint8), k3)
            n2, lab2 = cv2.connectedComponents(er, connectivity=8)
            ids2 = np.unique(lab2[(th > 0) & (er > 0)])
            ids2 = ids2[ids2 > 0]
            part = np.isin(lab2, ids2).astype(np.uint8)
            part = cv2.dilate(part, k3, iterations=2).astype(bool) & comp
            part[:lim_top] = False
            part[lim_bot + 1:] = False
            core |= part
        else:
            core |= comp
    ys, xs = np.nonzero(core)
    if len(ys) == 0:
        return None, None
    cy0, cy1, cx0, cx1 = ys.min(), ys.max(), xs.min(), xs.max()
    dist = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 3)
    core_set = set(core_ids.tolist())
    cand_ids = [i for i in range(1, n) if i not in core_set]
    m = core.copy()
    if cand_ids:
        mind = ndi.minimum(dist, lab, index=cand_ids)
        maxdt = ndi.maximum(dt[y0:y1], lab, index=cand_ids)
        ex = 3
        sat_bot = min(lim_bot, int(b + 0.1 * P) - y0)
        for i, md, mdt in zip(cand_ids, mind, maxdt):
            x, y, w, h, ar = st[i]
            if y < lim_top or y + h - 1 > sat_bot:
                continue
            if x < clip[0] or x + w - 1 >= W - clip[1]:
                continue
            inside = (x >= cx0 - ex and y >= cy0 - ex and x + w - 1 <= cx1 + ex and y + h - 1 <= cy1 + ex)
            if md <= 3 or (inside and md <= 14 and ar >= 15 and mdt >= 2.0) or (md <= 10 and ar >= 25 and mdt >= 2.5):
                m |= (lab == i)
    ys, xs = np.nonzero(m)
    bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    crop = m[by0:by1 + 1, bx0:bx1 + 1].astype(np.uint8)
    return crop, (bx0, by0 + y0, bx1, by1 + y0)
