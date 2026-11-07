"""Contact sheets of candidate crops from page pickles."""
from common import *
import pickle, sys

def load_rec(vol, pno):
    with open(os.path.join(WORK, 'cands', vol, f'{pno:04d}.pkl'), 'rb') as fh:
        return pickle.load(fh)

def unpack(c):
    h, w = c['crop_shape']
    return np.unpackbits(c['crop'], axis=1)[:, :w]

def make_sheet(imgs, labels=None, S=200, cols=12, path=None):
    rows = max(1, (len(imgs) + cols - 1) // cols)
    sheet = np.full((rows * (S + 22), cols * S), 170, np.uint8)
    for i, g in enumerate(imgs):
        gh, gw = g.shape
        sc = min(1.0, (S - 4) / max(gh, gw))
        g2 = cv2.resize(g, (max(1, int(gw * sc)), max(1, int(gh * sc))), interpolation=cv2.INTER_AREA) if sc < 1 else g
        r, cc = divmod(i, cols)
        y = r * (S + 22)
        sheet[y:y + g2.shape[0], cc * S:cc * S + g2.shape[1]] = g2
        if labels:
            cv2.putText(sheet, str(labels[i]), (cc * S + 2, y + S + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
    if path:
        cv2.imwrite(path, sheet)
    return sheet

if __name__ == '__main__':
    vol = sys.argv[1]
    for pno in [int(x) for x in sys.argv[2].split(',')]:
        rec = load_rec(vol, pno)
        imgs = [np.pad(((1 - unpack(c)) * 255).astype(np.uint8), 8, constant_values=255) for c in rec['cands']]
        labels = [f"{c['face']}{c['col']} {c['thick_area']}" for c in rec['cands']]
        make_sheet(imgs, labels, path=os.path.join(WORK, 'notes', f'sheet_{vol}_{pno}.png'))
        print(pno, len(imgs))
