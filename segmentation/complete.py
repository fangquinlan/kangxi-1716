"""Font-guided glyph completion.

For a matched headword the expected character is known. The font glyph is placed over the candidate
(template matching on the cleaned column ink at the volume's print scale), and every ink component that
lies mostly inside the dilated font shape is collected. Of the original mask, the font-assembled mask
and their union, the one most similar to the font is kept. This restores parts that the stroke-thickness
detector could not attach (thin-stroked tops such as 立/亠/宀, separated radicals) and drops attached junk.
"""
from detect import *
from feats import feature, tight
from repair import font_bin, font_feature
import detect as D

_ELL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))


class Strips:
    """Cleaned strips of one page, built on demand."""

    def __init__(self, doc, rec):
        self.doc, self.rec = doc, rec
        self.norm = None
        self.cache = {}

    def get(self, side, k):
        key = (side, k)
        if key not in self.cache:
            if self.norm is None:
                self.norm = normalize_bg(load_page(self.doc, self.rec['pno']))
            f = [f for f in self.rec['faces'] if f['side'] == side][0]
            P, thr = f['pitch'], self.rec['thr']
            strip, y0, xl = extract_strip(self.norm, f, k)
            raw = (strip < thr).astype(np.uint8)
            ink, dt, _ = prepare_strip(strip, thr)
            ink, _ = remove_rings(ink, P)
            ink, _ = remove_lines(ink, P)
            n, lab = cv2.connectedComponents(ink, connectivity=8)
            self.cache[key] = dict(strip=strip, y0=y0, xl=xl, ink=ink, lab=lab, forbid=raw & (1 - ink), P=P, thr=thr)
        return self.cache[key]


def place_font(ink, ch, scale, cy, cx, P):
    """Best placement (TM_CCOEFF_NORMED on blurred ink) of the scaled font glyph near (cy, cx).
    Returns (font mask in strip coords, score) or (None, -1)."""
    fb = font_bin(ch)
    if fb is None or fb.sum() == 0:
        return None, -1
    H, W = ink.shape
    best = (None, -1.0)
    y_lo, y_hi = int(max(0, cy - 0.75 * P)), int(min(H, cy + 0.75 * P))
    region = cv2.GaussianBlur(ink[y_lo:y_hi].astype(np.float32), (0, 0), 2)
    for sf in (0.94, 1.0, 1.06):
        th, tw = int(round(fb.shape[0] * scale * sf)), int(round(fb.shape[1] * scale * sf))
        if th < 8 or tw < 8 or th >= region.shape[0] or tw >= W:
            continue
        tb = cv2.resize(fb.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(region, cv2.GaussianBlur(tb, (0, 0), 2), cv2.TM_CCOEFF_NORMED)
        ys = np.arange(res.shape[0]) + y_lo + th / 2
        xs = np.arange(res.shape[1]) + tw / 2
        res[np.abs(ys - cy) > 0.3 * P, :] = -1
        res[:, np.abs(xs - cx) > 0.2 * P] = -1
        y, x = np.unravel_index(np.argmax(res), res.shape)
        s = float(res[y, x])
        if s > best[1]:
            m = np.zeros((H, W), np.uint8)
            m[y_lo + y:y_lo + y + th, x:x + tw] = (tb > 0.5)
            best = (m, s)
    return best


def assemble(lab, fmask, min_frac=0.6):
    """Union of ink components lying mostly (by pixel count) inside the dilated font shape."""
    foot = cv2.dilate(fmask, _ELL) > 0
    ids = np.unique(lab[foot])
    ids = ids[ids > 0]
    if len(ids) == 0:
        return None
    tot = np.bincount(lab.ravel(), minlength=lab.max() + 1)
    ins = np.bincount(lab[foot], minlength=lab.max() + 1)
    keep = ids[ins[ids] >= min_frac * tot[ids]]
    if len(keep) == 0:
        return None
    return np.isin(lab, keep)


def mask_sim(m, ch):
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return -1.0
    return float(feature(m[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)) @ font_feature(ch))


def complete(c, ch, S, scale):
    """c: candidate dict (strip-based). S: strip dict. Returns (new candidate dict, info)."""
    ink, lab, P = S['ink'], S['lab'], S['P']
    H, W = ink.shape
    bx0, by0, bx1, by1 = c['strip_box']
    h, w = c['crop_shape']
    m0 = np.zeros((H, W), bool)
    m0[by0:by1 + 1, bx0:bx1 + 1] = np.unpackbits(c['crop'], axis=1)[:, :w] > 0
    s0 = mask_sim(m0, ch)
    fm, tms = place_font(ink, ch, scale, (by0 + by1) / 2, (bx0 + bx1) / 2, P)
    info = dict(s0=s0, tm=tms, choice='orig', s=s0)
    if fm is None:
        return c, info
    mt = assemble(lab, fm)
    if mt is None:
        return c, info
    st = mask_sim(mt, ch)
    mu = m0 | mt
    su = mask_sim(mu, ch)
    info.update(st=st, su=su)
    if su >= s0 + 0.003 and su >= st - 0.01:
        m, info['choice'], info['s'] = mu, 'union', su
    elif st >= s0 + 0.02:
        m, info['choice'], info['s'] = mt, 'font', st
    else:
        return c, info
    ys, xs = np.nonzero(m)
    gx0, gy0, gx1, gy1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    crop = m[gy0:gy1 + 1, gx0:gx1 + 1].astype(np.uint8)
    qy0, qy1 = max(0, gy0 - 16), min(H, gy1 + 17)
    qx0, qx1 = max(0, gx0 - 16), min(W, gx1 + 17)
    fbm = S['forbid'][qy0:qy1, qx0:qx1]
    y0, xl = S['y0'], S['xl']
    nc = dict(c)
    nc.update(crop=np.packbits(crop, axis=1), crop_shape=crop.shape, strip_box=(gx0, gy0, gx1, gy1),
              gray=S['strip'][qy0:qy1, qx0:qx1].copy(), gray_off=(gx0 - qx0, gy0 - qy0),
              forbid=np.packbits(fbm, axis=1), forbid_shape=fbm.shape,
              y=int(gy0 + y0), yc=float((gy0 + gy1) / 2 + y0), x0=int(xl[gy0] + gx0), x1=int(xl[gy1] + gx1),
              w=int(gx1 - gx0 + 1), h=int(gy1 - gy0 + 1))
    return nc, info
