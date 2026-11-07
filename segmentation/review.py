"""Review sheets: scan crop next to font rendering."""
from common import *
from feats import tight
from PIL import Image, ImageDraw, ImageFont
import pickle, sys

FONT = os.path.join(ROOT, 'KangXiDict_Full-Regular.ttf')
_font = {}


def font(size):
    if size not in _font:
        _font[size] = ImageFont.truetype(FONT, size)
    return _font[size]


def render_char(ch, box=96):
    img = Image.new('L', (box * 2, box * 2), 255)
    d = ImageDraw.Draw(img)
    d.text((box // 2, box // 2), ch, fill=0, font=font(box))
    a = np.asarray(img)
    b = (a < 128).astype(np.uint8)
    t = tight(b)
    return ((1 - t) * 255).astype(np.uint8)


def fit(img, S):
    h, w = img.shape
    s = min((S - 4) / max(h, w), 1.0)
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def pair_sheet(items, path, S=110, cols=8):
    """items: list of (scan_gray_uint8, char or None, label)"""
    cw = 2 * S + 10
    rows = max(1, (len(items) + cols - 1) // cols)
    H = rows * (S + 20)
    sheet = Image.new('L', (cols * cw, H), 200)
    lab_font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 13) if os.path.exists('C:/Windows/Fonts/msyh.ttc') else None
    d = ImageDraw.Draw(sheet)
    for i, (scan, ch, label) in enumerate(items):
        r, c = divmod(i, cols)
        x, y = c * cw, r * (S + 20)
        a = fit(scan, S)
        sheet.paste(Image.fromarray(a), (x, y))
        if ch:
            b = fit(render_char(ch), S)
            sheet.paste(Image.fromarray(b), (x + S + 4, y))
        d.text((x + 2, y + S + 2), label, fill=0, font=lab_font)
    sheet.save(path)
