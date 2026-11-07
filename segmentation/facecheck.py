from pipeline import *
from concurrent.futures import ProcessPoolExecutor
def chk(a):
    vol, pno = a
    doc = fitz.open(pdf_list()[vol])
    im = load_page(doc, pno)
    norm = normalize_bg(im)
    thr, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (norm < thr).astype(np.uint8)
    faces = find_faces(ink)
    return vol, pno, [f['side'] for f in faces]
if __name__ == '__main__':
    jobs = []
    for vol in sorted(os.listdir(os.path.join(WORK,'cands'))):
        d = os.path.join(WORK,'cands',vol)
        for f in sorted(os.listdir(d)):
            rec = pickle.load(open(os.path.join(d,f),'rb'))
            if len(rec['faces']) < 2: jobs.append((vol, rec['pno']))
    still = []
    with ProcessPoolExecutor(10) as ex:
        for vol, pno, sides in ex.map(chk, jobs):
            if len(sides) < 2: still.append((vol, pno, sides))
    print(len(jobs), 'still missing', len(still))
    print(still)
