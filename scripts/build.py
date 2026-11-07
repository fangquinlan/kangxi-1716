"""Build complete CFF OpenType and WOFF2 from the archived original masks."""
import sys,json,pickle,time,zipfile,hashlib,argparse,os,ast,inspect,sqlite3,gc,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from kxfont.trace import trace,draw,library
from kxfont.optical import measure,fit_scales,place
from fontTools.pens.freetypePen import FreeTypePen
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.boundsPen import BoundsPen
from fontTools.fontBuilder import FontBuilder
from fontTools.ttLib import TTFont
from fontTools.cffLib import PrivateDict

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=1),encoding='utf-8')
def algorithm_key(config):
 # Compilation, placement and UI edits do not invalidate source traces.
 code=ast.dump(ast.parse(inspect.getsource(work)),include_attributes=False)
 code+=ast.dump(ast.parse(inspect.getsource(measure)),include_attributes=False)
 code+=ast.dump(ast.parse((ROOT/'src/kxfont/trace.py').read_text(encoding='utf-8')),include_attributes=False)
 settings={k:config[k] for k in ['trace_tolerance_px','trace_alpha']}
 return hashlib.sha256((code+json.dumps(settings,sort_keys=True)).encode()).hexdigest()
def entry_key(r,algo):
 return hashlib.sha256((algo+r['sha256']+str(r['pitch_px'])).encode()).hexdigest()
def work(item):
 r,config=item;path=Path(r['_override_file']) if '_override_file' in r else Path(os.environ.get('KX_GLYPHS',str(ROOT/'build/png')))/r['文件'];mask=np.asarray(Image.open(path).convert('L'))<128
 f=measure(mask,r['pitch_px']);paths=trace(mask,config['trace_tolerance_px'],config['trace_alpha'])
 # Source-resolution roundtrip checks every glyph. Thin source pixels and
 # counters are never dropped as turds. Retain a tighter trace if needed.
 def quality(paths):
  pen=FreeTypePen(None);draw(paths,pen);im=pen.image(width=mask.shape[1],height=mask.shape[0],transform=(1,0,0,-1,0,mask.shape[0]))
  aa=np.asarray(im.getchannel('A'))/255.;b=aa>=.5
  iou=float((mask&b).sum()/(mask|b).sum());area=float(abs(aa.sum()-mask.sum())/mask.sum())
  return iou,area
 iou,area=quality(paths);retry=False
 if iou<.965 or area>.025:
  alt=trace(mask,.08,.65);i2,a2=quality(alt)
  if i2>iou:paths,iou,area=alt,i2,a2;retry=True
 if iou<.95:
  alt=trace(mask,.02,0.0);i2,a2=quality(alt)
  if i2>iou:paths,iou,area=alt,i2,a2;retry=True
 f.update(iou=iou,area_error=area,trace_retry=retry,segments=sum(len(s[2]) for s in paths),contours=len(paths))
 return f,paths

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);parser.add_argument('--workers',type=int,default=8);a=parser.parse_args()
 config=read(ROOT/'font-config.json');all_rows=read(ROOT/'sources/manifest.json');by_id={r['id']:r for r in all_rows};rows=[r for r in all_rows if r['类别']=='字头'];rows.sort(key=lambda r:ord(r['字']))
 overrides_path=ROOT/'sources/overrides/manifest.json';overrides=read(overrides_path) if overrides_path.exists() else {}
 for r in rows:
  ov=overrides.get(f'{ord(r["字"]):05X}')
  if ov:
   op=ROOT/'sources/overrides'/ov['file']
   if hashlib.sha256(op.read_bytes()).hexdigest()!=ov['sha256']:raise ValueError('Override hash mismatch: '+r['字'])
   snapshot=ROOT/'build/inputs'/f'{ov["sha256"]}.png';snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(op.read_bytes())
   r['_override_file']=str(snapshot);r['sha256']=ov['sha256'];r['pitch_px']=ov['pitch_px']
   selected=by_id[ov['source_id']];r['source_id']=selected['id']
   for k in ['dpi_x','dpi_y','frame_mm_per_px']:r[k]=selected[k]
 if a.limit:rows=rows[:a.limit]
 if not os.environ.get('KX_GLYPHS') and not (ROOT/'build/png').exists():
  with zipfile.ZipFile(ROOT/'sources/glyphs-original.zip') as z:z.extractall(ROOT/'build/png')
 fingerprint=hashlib.sha256((ROOT/'font-config.json').read_bytes()+Path(__file__).read_bytes()+(ROOT/'src/kxfont/trace.py').read_bytes()+(ROOT/'src/kxfont/optical.py').read_bytes()+(ROOT/'sources/manifest.json').read_bytes()+json.dumps(overrides,sort_keys=True).encode()).hexdigest()
 algo=algorithm_key(config);keys=[entry_key(r,algo) for r in rows]
 db=sqlite3.connect(ROOT/'build/traces.sqlite3');db.execute('CREATE TABLE IF NOT EXISTS traces (key TEXT PRIMARY KEY, payload BLOB NOT NULL)')
 features=[];missing=[]
 for i,k in enumerate(keys):
  found=db.execute('SELECT payload FROM traces WHERE key=?',(k,)).fetchone()
  features.append(pickle.loads(found[0])[0] if found else None)
  if not found:missing.append(i)
 print('CACHE',len(rows)-len(missing),'RETRACE',len(missing),flush=True);t=time.time();library()
 with ThreadPoolExecutor(max_workers=a.workers) as ex:
  for j,(i,result) in enumerate(zip(missing,ex.map(work,[(rows[i],config) for i in missing]))):
   features[i]=result[0];db.execute('INSERT OR REPLACE INTO traces VALUES (?,?)',(keys[i],pickle.dumps(result,protocol=4)))
   if (j+1)%1000==0:db.commit();print('TRACE',j+1,len(missing),'seconds',round(time.time()-t),flush=True)
 db.commit();scales,scale_report=fit_scales(rows,features,config)
 glyphs={};hmtx={};vmtx={};cmap={};order=['.notdef','space','ideographicspace'];measurements=[];private=PrivateDict();private.nominalWidthX=0;private.defaultWidthX=0
 pen=T2CharStringPen(1000,None);pen.moveTo((150,50));pen.lineTo((850,50));pen.lineTo((850,750));pen.lineTo((150,750));pen.closePath();pen.moveTo((200,100));pen.lineTo((200,700));pen.lineTo((800,700));pen.lineTo((800,100));pen.closePath();glyphs['.notdef']=pen.getCharString();hmtx['.notdef']=(1000,150);vmtx['.notdef']=(1000,130)
 for name,advance in [('space',500),('ideographicspace',1000)]:glyphs[name]=T2CharStringPen(advance,None).getCharString();hmtx[name]=(advance,0);vmtx[name]=(1000,880)
 cmap.update({32:'space',160:'space',0x3000:'ideographicspace'})
 svgzip=zipfile.ZipFile(ROOT/'sources'/('glyphs-small.svg.zip' if a.limit else 'glyphs.svg.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=6)
 for i,(r,f,scale) in enumerate(zip(rows,features,scales)):
  paths=pickle.loads(db.execute('SELECT payload FROM traces WHERE key=?',(keys[i],)).fetchone()[0])[1]
  cp=ord(r['字']);name=f'u{cp:05X}';order.append(name);cmap[cp]=name
  source_bounds=BoundsPen(None);draw(paths,source_bounds);pl=place(f,scale,config,source_bounds.bounds)
  s,tx,ty=pl['scale'],pl['tx'],pl['ty'];trans=lambda x,y:(s*x+tx,880-(s*y+ty))
  pen=T2CharStringPen(1000,None);draw(paths,pen,trans);cs=pen.getCharString(private=private);glyphs[name]=cs
  bp=BoundsPen(None);cs.draw(bp);bounds=bp.bounds;hmtx[name]=(1000,round(bounds[0]));vmtx[name]=(1000,round(880-bounds[3]))
  cs.compile()
  sp=SVGPathPen(None,ntos=lambda v:format(v,'.2f').rstrip('0').rstrip('.') if v else '0');draw(paths,sp,lambda x,y:(s*x+tx,s*y+ty))
  svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000"><path fill="black" d="'+sp.getCommands()+'"/></svg>'
  svgzip.writestr(f'{name}.svg',svg)
  measurements.append({'id':r['id'],'source_id':r.get('source_id',r['id']),'char':r['字'],'cp':cp,'glyph':name,'source_sha256':r['sha256'],**f,**pl,'font_bounds':list(bounds),'width_mm_pdf':(f['bbox'][2]-f['bbox'][0])*25.4/r['dpi_x'],'height_mm_pdf':(f['bbox'][3]-f['bbox'][1])*25.4/r['dpi_y'],'height_mm_frame_estimate':(f['bbox'][3]-f['bbox'][1])*r['frame_mm_per_px']})
  if (i+1)%10000==0:print('COMPILE',i+1,len(rows),flush=True)
 svgzip.close();db.close();gc.collect()
 for alias,source in config['aliases'].items():
  if int(source,16) in cmap:cmap[int(alias,16)]=cmap[int(source,16)]
 fb=FontBuilder(1000,isTTF=False);fb.setupGlyphOrder(order);fb.setupCharacterMap(cmap);fb.setupCFF(config['postscript'],{'FullName':config['family']+' Regular','FamilyName':config['family'],'Weight':'Regular','version':config['version'],'Notice':config['copyright']},glyphs,{})
 fb.setupHorizontalMetrics(hmtx);fb.setupHorizontalHeader(ascent=880,descent=-120,lineGap=0);fb.setupVerticalMetrics(vmtx);fb.setupVerticalHeader(ascent=500,descent=-500,lineGap=0)
 license_text=(ROOT/'OFL.txt').read_text(encoding='utf-8')
 fb.setupNameTable({'familyName':config['family'],'styleName':'Regular','uniqueFontIdentifier':config['version']+';KX17;'+config['postscript'],'fullName':config['family']+' Regular','psName':config['postscript'],'version':'Version '+config['version']+'; Alpha','copyright':config['copyright'],'licenseDescription':license_text,'licenseInfoURL':'https://openfontlicense.org','description':'1716 historical glyphs; scan-derived outlines. Xuan retains its taboo missing stroke. Some glyphs retain scratch marks. Flagged-source review completed on 2026-11-08.'})
 for nid,val in [(1,config['family_zh']),(4,config['family_zh']+' 常规')]:fb.font['name'].setName(val,nid,3,1,0x804)
 fb.setupOS2(version=4,sTypoAscender=880,sTypoDescender=-120,sTypoLineGap=0,usWinAscent=880,usWinDescent=120,fsType=0,fsSelection=0x40|0x80,achVendID='KX17',sCapHeight=750,sxHeight=500)
 fb.setupPost(keepGlyphNames=False);fb.font['head'].created=fb.font['head'].modified=3876854400;fb.font['head'].fontRevision=float(config['version']);fb.font.recalcTimestamp=False
 # Reuse already decoded exact bounds instead of redrawing every CFF
 # outline separately for head, hhea, vhea and compression.
 bb={m['glyph']:m['font_bounds'] for m in measurements};bb['.notdef']=[150,50,850,750]
 bbox=[math.floor(min(v[0] for v in bb.values())),math.floor(min(v[1] for v in bb.values())),math.ceil(max(v[2] for v in bb.values())),math.ceil(max(v[3] for v in bb.values()))]
 fb.font['CFF '].cff.topDictIndex[0].FontBBox=bbox
 for field,val in zip(['xMin','yMin','xMax','yMax'],bbox):setattr(fb.font['head'],field,val)
 widths={n:math.ceil(v[2])-math.floor(v[0]) for n,v in bb.items()};heights={n:math.ceil(v[3])-math.floor(v[1]) for n,v in bb.items()}
 hh=fb.font['hhea'];hh.advanceWidthMax=1000;hh.minLeftSideBearing=min(hmtx[n][1] for n in bb);hh.minRightSideBearing=min(hmtx[n][0]-hmtx[n][1]-widths[n] for n in bb);hh.xMaxExtent=max(hmtx[n][1]+widths[n] for n in bb)
 vh=fb.font['vhea'];vh.advanceHeightMax=1000;vh.minTopSideBearing=min(vmtx[n][1] for n in bb);vh.minBottomSideBearing=min(vmtx[n][0]-vmtx[n][1]-heights[n] for n in bb);vh.yMaxExtent=max(vmtx[n][1]+heights[n] for n in bb)
 fb.font.recalcBBoxes=False
 name='OFLKangxi-Sample' if a.limit else 'OFLKangxi-Regular';otf=ROOT/'fonts'/(name+'.otf');fb.save(otf);print('OTF',otf.stat().st_size,flush=True)
 font=TTFont(otf,recalcTimestamp=False,recalcBBoxes=False);font.flavor='woff2';font.save(ROOT/'fonts'/(name+'.woff2'));print('WOFF2',(ROOT/'fonts'/(name+'.woff2')).stat().st_size,flush=True)
 prefix='sample-' if a.limit else ''
 dump(ROOT/'reports'/(prefix+'glyph-metrics.json'),measurements);dump(ROOT/'reports'/(prefix+'normalization.json'),scale_report)
 report={'glyphs':len(order),'source_headwords':len(rows),'cmap_entries':len(cmap),'otf_bytes':otf.stat().st_size,'woff2_bytes':(ROOT/'fonts'/(name+'.woff2')).stat().st_size,'trace_iou_percentiles':np.percentile([f['iou'] for f in features],[0,1,5,50,95,100]).tolist(),'trace_area_error_percentiles':np.percentile([f['area_error'] for f in features],[50,95,99,100]).tolist(),'trace_retries':sum(f['trace_retry'] for f in features),'size_safety_shrinks':sum(m['safety_shrink'] for m in measurements),'max_shift_units':max(max(abs(m['dx']),abs(m['dy'])) for m in measurements),'source_fingerprint':fingerprint,'potrace_version':library().potrace_version().decode(),'sha256':hashlib.sha256(otf.read_bytes()).hexdigest()}
 report['overrides_used']={k:{'sha256':v['sha256'],'char':v['char'],'source_id':v['source_id']} for k,v in overrides.items() if any(f'{ord(r["字"]):05X}'==k for r in rows)}
 dump(ROOT/'reports'/(prefix+'build-report.json'),report);print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
