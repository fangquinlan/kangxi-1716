"""Exercise edit persistence and rollback in an isolated temporary project."""
import importlib.util,tempfile,json,threading,urllib.request,urllib.error,hashlib,io
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('server',ROOT/'scripts/serve_review.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
rows=json.loads((ROOT/'sources/manifest.json').read_text(encoding='utf-8'));r=next(x for x in rows if x['字']=='𠈿' and x['类别']=='字头');s.ROWS={r['id']:r};s.PDFS={p.name.split('.')[1]:p for p in ROOT.parent.glob('康熙字典.*.pdf')}
with tempfile.TemporaryDirectory(dir=ROOT/'build',prefix='editor-test-') as tmp:
 s.ROOT=Path(tmp);(s.ROOT/'sources').mkdir();server=s.ThreadingHTTPServer(('127.0.0.1',0),s.Handler);s.PORT=server.server_address[1];threading.Thread(target=server.serve_forever,daemon=True).start();url=f'http://127.0.0.1:{s.PORT}'
 def post(path,data,origin=None):
  req=urllib.request.Request(url+path,data=json.dumps(data).encode(),headers={'Content-Type':'application/json','Origin':origin or url})
  with urllib.request.urlopen(req) as resp:return json.load(resp)
 data={'id':r['id'],'box':[int(r[k]) for k in ['页面x0','页面y0','页面x1','页面y1']]}
 preview=post('/api/preview',data);saved=post('/api/save',data);meta=saved['meta'];p=s.ROOT/'sources/overrides'/meta['file'];assert p.exists();assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256'];assert Image.open(p).mode=='1'
 manifest=json.loads((s.ROOT/'sources/overrides/manifest.json').read_text(encoding='utf-8'));assert f'{ord(r["字"]):05X}' in manifest
 post('/api/restore',{'id':r['id']});assert json.loads((s.ROOT/'sources/overrides/manifest.json').read_text(encoding='utf-8'))=={};assert len(list((s.ROOT/'sources/overrides/history').glob('*.png')))==1
 rejected=[]
 for payload,origin in [({**data,'box':[-1,0,10,10]},url),(data,'https://example.invalid')]:
  try:post('/api/save',payload,origin);raise AssertionError('bad request accepted')
  except urllib.error.HTTPError as e:rejected.append(e.code)
 assert rejected==[400,403];server.shutdown()
 result={'passed':True,'preview':True,'save_hash':True,'binary_png':True,'restore':True,'history_retained':True,'invalid_box_rejected':True,'foreign_origin_rejected':True,'real_overrides_untouched':True}
 (ROOT/'reports/editor-validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result)
