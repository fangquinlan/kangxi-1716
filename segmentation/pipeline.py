"""Stage 1: per-page candidate extraction. Saves one pickle per page with candidate crops."""
from detect import *
import pickle, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor

CAND_DIR = os.path.join(WORK, os.environ.get('KX_CANDS', 'cands'))


def reading_col(side, k):
    # right face: read from the 版心 side (k=7) to the gutter (k=0); left face: gutter (k=0) to 版心 (k=7)
    return 7 - k if side == 'R' else k


def thick_offsets(cands):
    """Thick-stroke centroid offset (in pitch units) of each candidate from its strip centre."""
    offs = []
    for c in cands:
        h, w = c['crop_shape']
        m = np.unpackbits(c['crop'], axis=1)[:, :w]
        closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        dt = cv2.distanceTransform(np.pad(closed, 2), cv2.DIST_L2, 5)[2:-2, 2:-2]
        ys, xs = np.nonzero(dt >= 4.0)
        if len(xs) >= 100:
            W = int(round(c['P'])) + 2 * MARGIN
            offs.append(((c['strip_box'][0] + xs.mean()) - W / 2) / c['P'])
    return np.array(offs)


def page_candidates(norm, thr):
    """Candidates of both faces. A face whose big glyphs are mostly half a column off-centre (faces printed
    without ruled lines can fool the column fit) is redone with the column grid shifted by half a pitch."""
    ink_full = (norm < thr).astype(np.uint8)
    faces = find_faces(ink_full)
    out, rings = [], []
    for f in faces:
        comb_fit(f, ink_full)
        c1, r1 = face_candidates(norm, thr, f)
        offs = thick_offsets(c1)
        if len(offs) >= 2 and (np.abs(offs) > 0.25).mean() >= 0.75:
            best = (np.abs(offs).mean(), f, c1, r1)
            for d in (-0.5, 0.5):
                g = dict(f)
                g['rules_u'] = [u + d * f['pitch'] for u in f['rules_u']]
                g['phase_shift'] = d
                c2, r2 = face_candidates(norm, thr, g)
                o2 = thick_offsets(c2)
                if len(o2) >= len(offs) * 0.75 and np.abs(o2).mean() < best[0]:
                    best = (np.abs(o2).mean(), g, c2, r2)
            _, f2, c1, r1 = best
            f.update(f2)
        out += c1
        rings += r1
    out.sort(key=lambda c: (c['face'], c['col'], c['yc']))
    return faces, out, rings


def face_candidates(norm, thr, f):
    out = []
    ring_list = []
    for f in [f]:
        P = f['pitch']
        fi = 0 if f['side'] == 'R' else 1
        for k in range(8):
            strip, y0, xl = extract_strip(norm, f, k)
            W = strip.shape[1]
            raw = (strip < thr).astype(np.uint8)
            ink, dt, (top, bot) = prepare_strip(strip, thr)
            ink, rings = remove_rings(ink, P)
            ink, caps = remove_lines(ink, P)
            closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            dt = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
            forbid = raw & (1 - ink)
            for (rcx, rcy, rr, rsc) in rings:
                ring_list.append(dict(face=fi, side=f['side'], k=k, col=reading_col(f['side'], k),
                                      yc=float(rcy + y0), x=float(xl[min(len(xl) - 1, int(rcy))] + rcx), r=rr, score=rsc))
            cl, cr = MARGIN - 8, MARGIN - 8
            # blank beyond the frame border (k=0, gutter side) and the 版心 line (k=7)
            if k == 0:
                if f['side'] == 'R':
                    cl = MARGIN - 2
                else:
                    cr = MARGIN - 2
            if k == 7:
                if f['side'] == 'R':
                    cr = MARGIN - 2
                else:
                    cl = MARGIN - 2
            cands, thick = detect_strip(ink, dt, P)
            for c in cands:
                crop, bb = glyph_mask(ink, thick, c, P, dt, clip=(cl, cr))
                if crop is None:
                    continue
                bx0, by0, bx1, by1 = bb
                w, h = bx1 - bx0 + 1, by1 - by0 + 1
                if max(w, h) < 0.3 * P:
                    continue
                # page coordinates (approx, rows are shifted by xl)
                X0 = int(xl[by0] + bx0)
                X1 = int(xl[by1] + bx1)
                # grayscale crop with 16px margin, from the rectified strip
                gy0, gy1 = max(0, by0 - 16), min(strip.shape[0], by1 + 17)
                gx0, gx1 = max(0, bx0 - 16), min(W, bx1 + 17)
                gray = strip[gy0:gy1, gx0:gx1].copy()
                fb = forbid[gy0:gy1, gx0:gx1]
                out.append(dict(face=fi, side=f['side'], k=k, col=reading_col(f['side'], k),
                                y=int(by0 + y0), yc=float((by0 + by1) / 2 + y0), x0=X0, x1=X1,
                                w=int(w), h=int(h), P=float(P), thick_area=int(c['thick_area']),
                                crop=np.packbits(crop, axis=1), crop_shape=crop.shape,
                                gray=gray, gray_off=(bx0 - gx0, by0 - gy0),
                                forbid=np.packbits(fb, axis=1), forbid_shape=fb.shape,
                                strip_box=(int(bx0), int(by0), int(bx1), int(by1)), strip_y0=int(y0)))
    return out, ring_list


def process_page(args):
    vol, pno = args
    path = os.path.join(CAND_DIR, vol, f'{pno:04d}.pkl')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return vol, pno, 'skip', 0
    try:
        doc = fitz.open(pdf_list()[vol])
        im = load_page(doc, pno)
        norm = normalize_bg(im)
        thr, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        faces, cands, rings = page_candidates(norm, thr)
        rec = dict(vol=vol, pno=pno, shape=im.shape, thr=float(thr),
                   faces=[{k: v for k, v in f.items()} for f in faces], cands=cands, rings=rings)
        tmp = path + '.tmp'
        with open(tmp, 'wb') as fh:
            pickle.dump(rec, fh, protocol=4)
        os.replace(tmp, path)
        return vol, pno, 'ok', len(cands)
    except Exception as e:
        return vol, pno, 'error: ' + repr(e) + traceback.format_exc()[-800:], 0


def main(vols, workers=10):
    jobs = []
    for vol in vols:
        os.makedirs(os.path.join(CAND_DIR, vol), exist_ok=True)
        n = fitz.open(pdf_list()[vol]).page_count
        jobs += [(vol, p) for p in range(n)]
    t = time.time()
    done = 0
    with ProcessPoolExecutor(workers) as ex:
        for vol, pno, st, n in ex.map(process_page, jobs, chunksize=2):
            done += 1
            if st not in ('ok', 'skip'):
                print(vol, pno, st, flush=True)
            if done % 100 == 0:
                print(f'{done}/{len(jobs)} {time.time() - t:.0f}s', flush=True)
    print('finished', len(jobs), f'{time.time() - t:.0f}s')


if __name__ == '__main__':
    main(sys.argv[1].split(','), int(sys.argv[2]) if len(sys.argv) > 2 else 10)
