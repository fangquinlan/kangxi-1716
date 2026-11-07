"""Glyph normalization / feature vectors shared by scans and font templates."""
import numpy as np, cv2

S = 40      # glyph box size inside the canvas
PAD = 4
SIGMA = 1.2
DIM = (S + 2 * PAD) ** 2


def norm_canvas(binary, S=S, pad=PAD):
    """binary: 2D array (1 = ink), tight crop. Aspect-preserving fit into S x S, centered."""
    h, w = binary.shape
    s = S / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    small = cv2.resize(binary.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((S + 2 * pad, S + 2 * pad), np.float32)
    y0 = pad + (S - nh) // 2
    x0 = pad + (S - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = small
    return canvas


def feature(binary, sigma=SIGMA):
    c = norm_canvas(binary)
    c = cv2.GaussianBlur(c, (0, 0), sigma)
    v = c.ravel()
    v = v - v.mean()
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v.astype(np.float32)


def tight(binary):
    ys, xs = np.nonzero(binary)
    if len(ys) == 0:
        return binary[:1, :1]
    return binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
