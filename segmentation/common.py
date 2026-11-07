import fitz, numpy as np, cv2, glob, os, json
ROOT = os.environ.get('KX_SOURCE_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sources', 'pdf')))
WORK = os.environ.get('KX_WORK', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build', 'segmentation')))
def pdf_list():
    fs = sorted(glob.glob(os.path.join(ROOT, '康熙字典.*.pdf')))
    return {os.path.basename(f).split('.')[1]: f for f in fs}
def load_page(doc, pno, gray=True):
    page = doc[pno]
    imgs = page.get_images(full=True)
    best = max(imgs, key=lambda i: i[2]*i[3])
    pix = doc.extract_image(best[0])
    buf = np.frombuffer(pix['image'], np.uint8)
    im = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    return im
