"""Final glyph rendering: clean 1-bit black-on-white image from a candidate record."""
import numpy as np, cv2


def unpack(c):
    h, w = c['crop_shape']
    return np.unpackbits(c['crop'], axis=1)[:, :w]


def clean_glyph(c, thr, grow_thr=22, grow_steps=3, hole_max=10, speck_min=6, margin=10):
    """c: candidate dict (crop mask + gray crop with offset). thr: page Otsu threshold on normalized gray.
    Returns uint8 image (0 = ink, 255 = paper) with `margin` white border."""
    mask = unpack(c).astype(np.uint8)
    gray = c['gray']
    ox, oy = c['gray_off']
    H, W = gray.shape
    seed = np.zeros((H, W), np.uint8)
    h, w = mask.shape
    seed[oy:oy + h, ox:ox + w] = mask
    # hysteresis: grow the glyph into slightly lighter ink connected to it (stroke edges, faint parts)
    weak = (gray < min(thr + grow_thr, 215)).astype(np.uint8)
    if 'forbid' in c:
        fh, fw = c['forbid_shape']
        fb = np.unpackbits(c['forbid'], axis=1)[:, :fw]
        weak &= 1 - cv2.dilate(fb, np.ones((3, 3), np.uint8))
    region = cv2.dilate(seed, np.ones((3, 3), np.uint8), iterations=grow_steps)
    g = seed.copy()
    k = np.ones((3, 3), np.uint8)
    for _ in range(grow_steps):
        g = cv2.dilate(g, k) & weak & region
        g |= seed
    # fill small holes (pinholes in strokes)
    inv = (1 - g).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, 4)
    for i in range(1, n):
        x, y, ww, hh, a = st[i]
        if a <= hole_max and x > 0 and y > 0 and x + ww < W and y + hh < H:
            g[lab == i] = 1
    # remove isolated specks
    n, lab, st, _ = cv2.connectedComponentsWithStats(g, 8)
    for i in range(1, n):
        if st[i, 4] < speck_min:
            g[lab == i] = 0
    ys, xs = np.nonzero(g)
    if len(ys) == 0:
        return None
    g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    out = np.full((g.shape[0] + 2 * margin, g.shape[1] + 2 * margin), 255, np.uint8)
    out[margin:-margin, margin:-margin][g > 0] = 0
    return out


def save_1bit(img, path):
    from PIL import Image
    Image.fromarray(img).convert('1').save(path, optimize=True)
