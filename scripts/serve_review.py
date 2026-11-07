"""Loopback review server: source-page rectangle -> reversible glyph override."""
import argparse,base64,hashlib,io,json,os,threading,time,urllib.parse
from functools import lru_cache
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
import fitz,numpy as np,cv2
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
LOCK=threading.Lock();ROWS={};PDFS={};PORT=8766
def target_cp(r):
 cp=f'{ord(r["字"]):05X}';aliases=json.loads((ROOT/'font-config.json').read_text(encoding='utf-8'))['aliases'] if (ROOT/'font-config.json').exists() else {'7384':'248E5'}
 return f'{int(aliases.get(cp.lstrip("0"),cp),16):05X}'
def overrides():
 p=ROOT/'sources/overrides/manifest.json'
 return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
@lru_cache(maxsize=4)
def page(v,p):
 with fitz.open(PDFS[v]) as doc:
  best=max(doc[p-1].get_images(full=True),key=lambda i:i[2]*i[3]);raw=doc.extract_image(best[0])['image']
 return Image.open(io.BytesIO(raw)).convert('RGB')
def png_bytes(im):
 b=io.BytesIO();im.save(b,format='PNG',optimize=True);return b.getvalue()
def recut(data):
 r=ROWS[data['id']];im=page(r['册'],int(r['PDF页']));box=[int(round(float(v))) for v in data['box']]
 if len(box)!=4:raise ValueError('需要四个坐标')
 x0,y0,x1,y1=box
 if not(0<=x0<x1<=im.width and 0<=y0<y1<=im.height):raise ValueError('字框超出原页范围')
 if x1-x0<6 or y1-y0<6 or x1-x0>700 or y1-y0>700:raise ValueError('请选择6至700像素的字区')
 gray=np.asarray(im.crop(box).convert('L'))
 # Background estimate only changes threshold measurement, not stroke geometry.
 k=max(15,min(81,(min(gray.shape)//3)|1));bg=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)));bg=cv2.GaussianBlur(bg,(0,0),3);norm=cv2.divide(gray,bg,scale=255)
 threshold=data.get('threshold')
 if threshold is None:threshold=float(cv2.threshold(norm,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0])
 threshold=float(threshold)
 if not 40<=threshold<=245:raise ValueError('阈值须在40至245之间')
 mask=norm<threshold
 if not mask.any():raise ValueError('框内没有墨迹')
 yy,xx=np.nonzero(mask);tight=mask[yy.min():yy.max()+1,xx.min():xx.max()+1];out=np.full((tight.shape[0]+20,tight.shape[1]+20),255,np.uint8);out[10:-10,10:-10][tight]=0
 binary=png_bytes(Image.fromarray(out).convert('1'));meta={'source_id':r['id'],'char':r['字'],'source_sha256':r['sha256'],'volume':r['册'],'page':int(r['PDF页']),'box':box,'tight_page_box':[x0+int(xx.min()),y0+int(yy.min()),x0+int(xx.max())+1,y0+int(yy.max())+1],'threshold':threshold,'algorithm':'local-background-otsu-v1; no hole fill; no component deletion','sha256':hashlib.sha256(binary).hexdigest(),'width':out.shape[1],'height':out.shape[0],'pitch_px':r['pitch_px']}
 return binary,meta
class Handler(SimpleHTTPRequestHandler):
 def __init__(self,*a,**kw):super().__init__(*a,directory=str(ROOT),**kw)
 def log_message(self,*a):pass
 def send(self,status,body,ctype='application/json; charset=utf-8'):
  if not isinstance(body,bytes):body=json.dumps(body,ensure_ascii=False).encode()
  self.send_response(status);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
 def do_GET(self):
  u=urllib.parse.urlparse(self.path);q=urllib.parse.parse_qs(u.query)
  try:
   if u.path=='/api/status':return self.send(200,{'editable':True,'project':'OFL Kangxi'})
   if u.path=='/api/glyph':
    r=ROWS[q['id'][0]];im=page(r['册'],int(r['PDF页']));ov=overrides().get(target_cp(r));original=[int(r[k]) for k in ['页面x0','页面y0','页面x1','页面y1']]
    return self.send(200,{'id':r['id'],'char':r['字'],'target_cp':target_cp(r),'page':r['PDF页'],'volume':r['集'],'size':list(im.size),'original_box':original,'box':ov['box'] if ov and ov['source_id']==r['id'] else original,'override':ov})
   if u.path=='/api/page':
    r=ROWS[q['id'][0]];im=page(r['册'],int(r['PDF页']));b=io.BytesIO();im.save(b,format='JPEG',quality=94);return self.send(200,b.getvalue(),'image/jpeg')
   if u.path=='/api/overrides':
    return self.send(200,overrides())
   if u.path.startswith('/api/'):return self.send(404,{'error':'接口不存在'})
   decoded=urllib.parse.unquote(u.path);parts=decoded.strip('/').split('/')
   if parts[0] in ['build','.git'] or '..' in parts:return self.send(403,{'error':'禁止访问'})
   return super().do_GET()
  except (KeyError,ValueError,OSError) as e:return self.send(400,{'error':str(e)})
 def do_POST(self):
  try:
   origin=self.headers.get('Origin');allowed={f'http://127.0.0.1:{PORT}',f'http://localhost:{PORT}'}
   if origin not in allowed: return self.send(403,{'error':'只接受同源校勘页请求'})
   if not self.headers.get('Content-Type','').startswith('application/json'):return self.send(415,{'error':'需要JSON'})
   n=int(self.headers.get('Content-Length','0'))
   if n<1 or n>20000:return self.send(413,{'error':'请求过大'})
   data=json.loads(self.rfile.read(n))
   if self.path not in ['/api/preview','/api/save','/api/restore']:return self.send(404,{'error':'接口不存在'})
   if self.path=='/api/restore':
    cp=target_cp(ROWS[data['id']]);p=ROOT/'sources/overrides/manifest.json'
    with LOCK:
     entries=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {};entries.pop(cp,None);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(entries,ensure_ascii=False,indent=2),encoding='utf-8')
    return self.send(200,{'restored':True,'codepoint':cp})
   binary,meta=recut(data)
   if self.path=='/api/save':
    cp=target_cp(ROWS[data['id']]);meta['target_cp']=cp;folder=ROOT/'sources/overrides';folder.mkdir(exist_ok=True);(folder/'history').mkdir(exist_ok=True);p=folder/'manifest.json'
    meta['saved_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z');meta['note']=str(data.get('note',''))[:2000];meta['file']=f'png/u{cp}.png';(folder/'png').mkdir(exist_ok=True)
    with LOCK:
     entries=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
     stamp=str(time.time_ns());(folder/'history'/f'u{cp}-{stamp}.png').write_bytes(binary);(folder/'history'/f'u{cp}-{stamp}.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
     (folder/meta['file']).write_bytes(binary);entries[cp]=meta;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(entries,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,p)
   return self.send(200,{'image':'data:image/png;base64,'+base64.b64encode(binary).decode(),'meta':meta,'saved':self.path=='/api/save'})
  except (KeyError,ValueError,OSError) as e:return self.send(400,{'error':str(e)})
def main():
 global ROWS,PDFS,PORT
 p=argparse.ArgumentParser();p.add_argument('--pdf-root',type=Path,default=ROOT.parent);p.add_argument('--port',type=int,default=8766);a=p.parse_args();PORT=a.port
 ROWS={r['id']:r for r in json.loads((ROOT/'sources/manifest.json').read_text(encoding='utf-8'))};PDFS={p.name.split('.')[1]:p for p in a.pdf_root.glob('康熙字典.*.pdf')}
 print(f'Kangxi review: http://127.0.0.1:{PORT}/proof/review.html | PDFs: {len(PDFS)}',flush=True);ThreadingHTTPServer(('127.0.0.1',PORT),Handler).serve_forever()
if __name__=='__main__':main()
