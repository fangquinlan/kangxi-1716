"""Meaningful structural, all-glyph raster and encoded-metric checks."""
import json,sys,hashlib,zipfile,argparse,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import numpy as np,freetype
from fontTools.ttLib import TTFont
from fontTools.pens.boundsPen import BoundsPen
def main():
 p=argparse.ArgumentParser();p.add_argument('--sample',action='store_true');args=p.parse_args();prefix='sample-' if args.sample else '';stem='OFLKangxi-Sample' if args.sample else 'OFLKangxi-Regular';path=ROOT/'fonts'/(stem+'.otf')
 tt=TTFont(path,checkChecksums=2,recalcTimestamp=False,recalcBBoxes=False);w=TTFont(ROOT/'fonts'/(stem+'.woff2'),recalcTimestamp=False,recalcBBoxes=False);cmap=tt.getBestCmap();metrics=json.loads((ROOT/'reports'/(prefix+'glyph-metrics.json')).read_text(encoding='utf-8'));errors=[]
 expect={m['cp'] for m in metrics};assert expect<=set(cmap)
 assert tt['maxp'].numGlyphs==len(metrics)+3
 assert tt['OS/2'].fsType==0
 assert 'vmtx' in tt and 'vhea' in tt
 table_checks={tag:tt.getTableData(tag)==w.getTableData(tag) for tag in tt.keys() if tag not in ['GlyphOrder','head']};assert all(table_checks.values()),table_checks
 face=freetype.Face(str(path));face.set_pixel_sizes(0,32);t=time.time();empty=[];bad_bounds=[];outliers=[];gs=tt.getGlyphSet()
 for i,m in enumerate(metrics):
  name=cmap[m['cp']];assert tt['hmtx'][name][0]==1000;assert tt['vmtx'][name][0]==1000
  face.load_char(m['char'],freetype.FT_LOAD_RENDER|freetype.FT_LOAD_NO_HINTING);bm=face.glyph.bitmap
  if not bm.width or not bm.rows or not any(bm.buffer):empty.append(m['char'])
  bp=BoundsPen(gs);gs[name].draw(bp);x0,y0,x1,y1=bp.bounds
  if x0<29.99 or x1>970.01 or y0<-90.01 or y1>850.01:bad_bounds.append(m['char'])
  assert np.allclose(bp.bounds,m['font_bounds'],atol=.01),(m['char'],bp.bounds,m['font_bounds'])
  if m['iou']<.94 or m['area_error']>.04:outliers.append({'char':m['char'],'cp':m['cp'],'iou':m['iou'],'area_error':m['area_error']})
  if i%10000==0:print('VALIDATE',i,len(metrics),flush=True)
 assert not empty,empty;assert not bad_bounds,bad_bounds
 if not args.sample:assert cmap[0x7384]==cmap[0x248E5]
 raster={};chars='一二三丨丿丶乙人小口大山川田國鬱龜龍麤爨𠈿𠯉玄𤣥'
 for size in [16,24,32,48,96,256]:
  face.set_pixel_sizes(0,size);n=0
  for ch in chars:
   if ord(ch) not in cmap:continue
   face.load_char(ch,freetype.FT_LOAD_RENDER|freetype.FT_LOAD_NO_HINTING);assert any(face.glyph.bitmap.buffer);n+=1
  raster[size]=n
 # Compare archive source hashes; originals include duplicates and headings.
 source_manifest=json.loads((ROOT/'sources/manifest.json').read_text(encoding='utf-8'));hash_checked=0
 with zipfile.ZipFile(ROOT/'sources/glyphs-original.zip') as z:
  for r in source_manifest:
   assert hashlib.sha256(z.read(r['文件'])).hexdigest()==r['sha256'];hash_checked+=1
 import uharfbuzz as hb
 font=hb.Font(hb.Face(path.read_bytes()));font.scale=(1000,1000);shaping={}
 text=''.join(ch for ch in chars if ord(ch) in cmap) or ''.join(m['char'] for m in metrics[:12])
 for direction in ['ltr','ttb']:
  buf=hb.Buffer();buf.add_str(text);buf.direction=direction;buf.script='hani';buf.language='zh';hb.shape(font,buf,{})
  assert len(buf.glyph_infos)==len(text)
  assert all(i.codepoint!=0 for i in buf.glyph_infos)
  expected=(1000,0) if direction=='ltr' else (0,-1000)
  assert all((p.x_advance,p.y_advance)==expected for p in buf.glyph_positions)
  shaping[direction]={'glyphs':len(text),'advance':expected}
 result={'passed':True,'glyphs':len(metrics),'source_archive_hashes_checked':hash_checked,'empty_rasters':empty,'bounds_failures':bad_bounds,'trace_review_outliers':outliers,'lossless_woff2_tables':table_checks,'multi_size_probe_count':raster,'harfbuzz':shaping,'xuan_alias_verified':not args.sample,'elapsed_seconds':round(time.time()-t,2),'otf_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'limitations':['Geometric metrics are not manual proofreading or a reading preference study.','Low-size raster coverage is all glyphs at 32px; named probes at six sizes.']}
 (ROOT/'reports'/(prefix+'validation.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='trace_review_outliers'},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
