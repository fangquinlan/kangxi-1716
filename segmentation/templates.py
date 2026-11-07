"""Render font templates for every JSON headword and compute features."""
from common import *
from feats import *
from PIL import Image, ImageDraw, ImageFont
import time

FONT = os.environ.get('KX_REFERENCE_FONT', os.path.join(ROOT, 'reference.ttf'))
JSON = os.environ.get('KX_HEADWORDS', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sources', 'headwords.json')))
OUT = os.path.join(WORK, 'templates.npz')
SIZE = 128


def render(ch, font):
    img = Image.new('L', (SIZE * 2, SIZE * 2), 0)
    d = ImageDraw.Draw(img)
    d.text((SIZE // 2, SIZE // 2), ch, fill=255, font=font)
    a = np.asarray(img)
    return (a > 127).astype(np.uint8)


def main():
    d = json.load(open(JSON, encoding='utf-8'))
    chars = []
    seen = set()
    for p in sorted(d['pages'], key=int):
        for ch in d['pages'][p]:
            if ch not in seen:
                seen.add(ch)
                chars.append(ch)
    font = ImageFont.truetype(FONT, SIZE)
    t = time.time()
    feats = np.zeros((len(chars), DIM), np.float16)
    aspect = np.zeros(len(chars), np.float32)
    ok = np.zeros(len(chars), bool)
    for i, ch in enumerate(chars):
        b = render(ch, font)
        if b.sum() == 0:
            continue
        tb = tight(b)
        feats[i] = feature(tb)
        aspect[i] = tb.shape[1] / tb.shape[0]
        ok[i] = True
        if i % 5000 == 0:
            print(i, time.time() - t, flush=True)
    np.savez_compressed(OUT, chars=np.array(chars), feats=feats, aspect=aspect, ok=ok)
    print('done', len(chars), ok.sum(), time.time() - t)


if __name__ == '__main__':
    main()
