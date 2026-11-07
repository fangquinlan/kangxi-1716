"""Font-guided cleanup of a binary headword glyph.

The reference font was built from the same dictionary glyphs, so after registering the font glyph onto the
scan, ink that lies clearly outside the font's strokes and is thin (box edges, rule pieces, stray text) is
removed. Thick parts, and short thin parts such as stroke tips, are never removed.
"""
import numpy as np, cv2, os
from PIL import Image, ImageDraw, ImageFont

FONT = os.environ.get('KX_REFERENCE_FONT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sources', 'reference.ttf')))
_fonts = {}


def font_glyph(ch, size):
    """Tight binary rendering (1 = ink) of ch at the given pixel size."""
    if size not in _fonts:
        _fonts[size] = ImageFont.truetype(FONT, size)
    f = _fonts[size]
    img = Image.new('L', (size * 2, size * 2), 0)
    ImageDraw.Draw(img).text((size // 2, size // 2), ch, fill=255, font=f)
    a = (np.asarray(img) > 127).astype(np.uint8)
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return None
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def core_box(g):
    """Bounding box of the thick parts (robust against thin junk)."""
    closed = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    dt = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
    ys, xs = np.nonzero(dt >= 3.0)
    if len(ys) < 20:
        ys, xs = np.nonzero(g)
    return xs.min(), ys.min(), xs.max(), ys.max()


def _place(g, f, box, pad):
    G = np.zeros((g.shape[0] + 2 * pad, g.shape[1] + 2 * pad), np.float32)
    bx0, by0, bx1, by1 = box
    cy, cx = (by0 + by1) / 2 + pad, (bx0 + bx1) / 2 + pad
    fy, fx = f.shape
    oy, ox = int(round(cy - fy / 2)), int(round(cx - fx / 2))
    ya, xa = max(0, oy), max(0, ox)
    yb, xb = min(G.shape[0], oy + fy), min(G.shape[1], ox + fx)
    if yb > ya and xb > xa:
        G[ya:yb, xa:xb] = f[ya - oy:yb - oy, xa - ox:xb - ox]
    return G


def register(g, ch, pad=24):
    """Register the font glyph onto binary glyph g (1 = ink). Returns (font image in g-coords [float 0..1],
    correlation) or (None, 0). Tries a placement fitted to the thick core and one fitted to the full bbox."""
    H, W = g.shape
    f0 = font_glyph(ch, 128)
    if f0 is None:
        return None, 0.0
    G = np.pad(g.astype(np.float32), pad)
    Gb = cv2.GaussianBlur(G, (0, 0), 2.5)
    ys, xs = np.nonzero(g)
    boxes = [core_box(g), (xs.min(), ys.min(), xs.max(), ys.max())]
    best = (None, -1.0)
    for box in boxes:
        bx0, by0, bx1, by1 = box
        sh = (by1 - by0 + 1) / f0.shape[0]
        sw = (bx1 - bx0 + 1) / f0.shape[1]
        size = int(round(128 * (sh + sw) / 2 * 1.02))
        size = max(40, min(400, size))
        f = font_glyph(ch, size)
        if f is None:
            continue
        tmpl = _place(g, f, box, pad)
        Tb = cv2.GaussianBlur(tmpl, (0, 0), 2.5)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-4)
            cc, warp = cv2.findTransformECC(Gb, Tb, warp, cv2.MOTION_AFFINE, crit, None, 5)
        except cv2.error:
            cc = -1.0
        A = warp[:, :2]
        sx, sy = np.linalg.norm(A[:, 0]), np.linalg.norm(A[:, 1])
        if cc < 0 or not (0.8 < sx < 1.25 and 0.8 < sy < 1.25) or abs(warp[0, 2]) > 30 or abs(warp[1, 2]) > 30:
            warp = np.eye(2, 3, dtype=np.float32)
            cc = float(np.corrcoef(Gb.ravel(), Tb.ravel())[0, 1])
        if cc > best[1]:
            aligned = cv2.warpAffine(tmpl, warp, (G.shape[1], G.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            best = (aligned[pad:pad + H, pad:pad + W], float(cc))
    return best


def line_like(comp_mask, max_thick=6.5, min_len=22, max_angle=9.0, min_elong=3.5):
    """Straight, thin, axis-aligned piece (box edge / rule), as opposed to a tapering diagonal stroke tip."""
    ys, xs = np.nonzero(comp_mask)
    if len(ys) < 8:
        return False, 0.0, 0.0, 0.0
    pts = np.stack([xs, ys], 1).astype(np.float32)
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    length, thick = max(w, h), min(w, h)
    # orientation of the long side, folded to [0, 90): 0 = horizontal, 90 = vertical
    if w < h:
        ang = ang + 90
    ang = abs(((ang + 90) % 180) - 90)
    axis_dev = min(ang, abs(90 - ang))
    ok = length >= min_len and thick <= max_thick and axis_dev <= max_angle and length >= min_elong * max(thick, 1.0)
    # long, uniformly thin straight lines at any angle (woodblock cracks, scratches)
    ok = ok or (length >= 35 and thick <= 3.5 and length >= 8 * max(thick, 1.0))
    return ok, float(length), float(thick), float(axis_dev)


def extraneous(g, fimg, r=6, thin_dt=2.6):
    """Thin, straight, axis-aligned ink of g lying more than r px from the registered font strokes."""
    support = cv2.dilate((fimg > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    out = (g > 0) & (support == 0)
    closed = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    dt = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
    n, lab, st, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8), 8)
    rem = np.zeros(g.shape, bool)
    pieces = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        comp = lab == i
        mdt = float(dt[comp].max())
        if mdt > thin_dt:
            continue
        ok, length, thick, dev = line_like(comp)
        if ok:
            rem |= comp
            pieces.append((int(x), int(y), int(w), int(h), round(length), round(thick, 1), round(dev, 1)))
    return rem, dict(removed=int(rem.sum()), total=int(g.sum()), pieces=pieces)


def clean_with_font(img, ch, margin=10):
    """img: uint8 rendered glyph (0 ink / 255 paper, with margin). Returns cleaned image and stats."""
    g = (img < 128).astype(np.uint8)
    fimg, cc = register(g, ch)
    if fimg is None or cc < 0.85:
        return img, dict(removed=0, cc=cc, skipped=True)
    rem, stt = extraneous(g, fimg)
    stt['cc'] = cc
    if stt['removed'] == 0:
        return img, stt
    g = g & ~rem
    ys, xs = np.nonzero(g)
    g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    out = np.full((g.shape[0] + 2 * margin, g.shape[1] + 2 * margin), 255, np.uint8)
    out[margin:-margin, margin:-margin][g > 0] = 0
    return out, stt
