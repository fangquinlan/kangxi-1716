"""Detect/remove long straight hairlines (scratches, woodblock cracks, rule remnants) crossing a glyph.

A hairline is a straight segment, at most ~3 px wide, at least 35% of the glyph's size long, that is
inked along most of its length. Only its thin pixels are removed; pixels inside real (thick) strokes stay.
"""
import numpy as np, cv2


def thin_mask(g):
    """Ink belonging to structures narrower than ~5 px and not touching a wider stroke."""
    e5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    wide = cv2.morphologyEx(g, cv2.MORPH_OPEN, e5)
    wide = cv2.dilate(wide, np.ones((3, 3), np.uint8))
    return ((g > 0) & (wide == 0)).astype(np.uint8)


def find_hairlines(g, min_frac=0.25, max_gap=24):
    """g: binary glyph (1 = ink), tight. Returns list of segments (x1, y1, x2, y2, coverage, thin_frac)."""
    H, W = g.shape
    thin = thin_mask(g)
    L = int(min_frac * max(H, W))
    segs = cv2.HoughLinesP(thin * 255, 1, np.pi / 360, threshold=max(20, L // 3), minLineLength=L, maxLineGap=max_gap)
    out = []
    if segs is None:
        return out
    for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4):
        n = int(max(abs(x2 - x1), abs(y2 - y1))) + 1
        xs = np.round(np.linspace(x1, x2, n)).astype(int)
        ys = np.round(np.linspace(y1, y2, n)).astype(int)
        on = g[ys, xs] > 0
        th = thin[ys, xs] > 0
        cov = on.mean()
        thin_frac = th.mean()
        ang = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
        # Song-style strokes include thin horizontals and thin rising '/' strokes (提, e.g. in 氵), but no thin
        # straight '\' or vertical strokes: only lines falling to the right or (near-)vertical qualify
        falling = (x2 - x1) * (y2 - y1) > 0
        if ang < 20 or not ((falling and ang <= 87) or ang >= 88):
            continue
        if cov >= 0.85 and thin_frac >= 0.5:
            out.append((int(x1), int(y1), int(x2), int(y2), float(cov), float(thin_frac), float(ang)))
    return out


def remove_hairlines(g, segs, width=2.0):
    """Remove thin pixels lying within `width` px of any hairline segment; returns cleaned glyph and count."""
    if not segs:
        return g, 0
    H, W = g.shape
    m = np.zeros((H, W), np.uint8)
    for x1, y1, x2, y2, *_ in segs:
        # extend slightly beyond the detected ends
        dx, dy = x2 - x1, y2 - y1
        n = np.hypot(dx, dy) or 1
        ex, ey = dx / n * 6, dy / n * 6
        cv2.line(m, (int(round(x1 - ex)), int(round(y1 - ey))), (int(round(x2 + ex)), int(round(y2 + ey))), 1,
                 thickness=int(2 * width + 1))
    rem = (m > 0) & (thin_mask(g) > 0)
    g2 = g.copy()
    g2[rem] = 0
    # drop specks the removal leaves behind
    n2, lab, st, _ = cv2.connectedComponentsWithStats(g2, 8)
    for i in range(1, n2):
        if st[i, 4] < 12:
            comp = lab == i
            if (comp & (cv2.dilate(m, np.ones((5, 5), np.uint8)) > 0)).any():
                g2[comp] = 0
    return g2, int((g > 0).sum() - (g2 > 0).sum())
