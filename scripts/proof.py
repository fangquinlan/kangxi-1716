"""Build portable review cards and horizontal/vertical font specimens."""
import json,base64,io,sys,argparse,collections,concurrent.futures,os
from pathlib import Path
from PIL import Image,ImageDraw
import fitz
ROOT=Path(__file__).resolve().parents[1]
def context(job):
 pdf,pno,rows=job;doc=fitz.open(pdf);pg=doc[pno];im=max(pg.get_images(full=True),key=lambda i:i[2]*i[3]);a=Image.open(io.BytesIO(doc.extract_image(im[0])['image'])).convert('RGB');doc.close()
 for r in rows:
  dst=ROOT/'proof/context'/(r['id']+'.jpg')
  if dst.exists():continue
  x0,y0,x1,y1=[int(r[k]) for k in ['页面x0','页面y0','页面x1','页面y1']];pad=95
  bb=(max(0,x0-pad),max(0,y0-pad),min(a.width,x1+pad),min(a.height,y1+pad));c=a.crop(bb);dr=ImageDraw.Draw(c);dr.rectangle((x0-bb[0],y0-bb[1],x1-bb[0],y1-bb[1]),outline='#bf5b25',width=2);c.thumbnail((470,470));c.save(dst,quality=85,optimize=True)
 return len(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--pdf-root',type=Path);args=p.parse_args()
 rows=json.loads((ROOT/'reports/review-items.json').read_text(encoding='utf-8'));(ROOT/'proof/context').mkdir(exist_ok=True)
 if args.pdf_root:
  pdfs={p.name.split('.')[1]:p for p in args.pdf_root.glob('康熙字典.*.pdf')};groups=collections.defaultdict(list)
  for r in rows:
   if not (ROOT/'proof/context'/(r['id']+'.jpg')).exists():groups[(r['册'],int(r['PDF页'])-1)].append(r)
  jobs=[(pdfs[v],p,rs) for (v,p),rs in groups.items()]
  with concurrent.futures.ThreadPoolExecutor(4) as ex:
   for i,n in enumerate(ex.map(context,jobs)):
    if (i+1)%100==0:print('CONTEXT',i+1,len(jobs),flush=True)
 for r in rows:
  r['png']='data:image/png;base64,'+base64.b64encode((Path(os.environ.get('KX_GLYPHS',str(ROOT/'build/png')))/r['文件']).read_bytes()).decode()
 data=json.dumps(rows,ensure_ascii=False,separators=(',',':'));(ROOT/'proof/review-data.js').write_text('const REVIEW='+data+';',encoding='utf-8')
 print('PROOF ROWS',len(rows),flush=True)
if __name__=='__main__':main()
