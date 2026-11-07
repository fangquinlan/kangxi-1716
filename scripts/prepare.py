"""Import original masks and recover scanner/layout geometry without resampling."""
import argparse,csv,json,pickle,hashlib,zipfile,shutil,collections
from pathlib import Path
import fitz
ROOT=Path(__file__).resolve().parents[1]
def dump(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=1),encoding='utf-8')
def main():
 p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--pdf-root',type=Path,required=True);a=p.parse_args()
 rows=list(csv.DictReader((a.original/'字头清单.csv').open(encoding='utf-8-sig',newline='')))
 hair=json.loads((a.work/'hairline_applied.json').read_text(encoding='utf-8'));frag=json.loads((a.work/'detached_changes.json').read_text(encoding='utf-8'))
 pdfs={p.name.split('.')[1]:p for p in a.pdf_root.glob('康熙字典.*.pdf')};by=collections.defaultdict(list)
 for i,r in enumerate(rows):r['id']=f'kx{i+1:05d}';by[(r['册'],int(r['PDF页'])-1)].append(r)
 counts=collections.Counter();geometry=[];review=[]
 for vol in sorted(pdfs):
  doc=fitz.open(pdfs[vol])
  for (v,pno),rr in sorted(by.items()):
   if v!=vol:continue
   rec=pickle.load(open(a.work/'cands'/v/f'{pno:04d}.pkl','rb'));pg=doc[pno]
   im=max(pg.get_images(full=True),key=lambda i:i[2]*i[3]);rect=pg.get_image_bbox(im);dpi=[im[2]*72/rect.width,im[3]*72/rect.height]
   faces={f['side']:f for f in rec['faces']};geometry.append({'volume':v,'page':pno+1,'scan_px':rec['shape'][::-1],'pdf_dpi':dpi,'faces':list(faces.values())})
   for r in rr:
    f=faces['R' if r['面']=='右' else 'L'];r['pitch_px']=f['pitch'];r['frame_height_px']=f['ybot']-f['ytop'];r['dpi_x'],r['dpi_y']=dpi;r['frame_mm_per_px']=193/r['frame_height_px']
    tags=[]
    if r['类别']!='部首标题' and float(r['字形相似度'] or 0)<.6:tags.append('低相似度')
    if r['来源']:tags.append(r['来源'])
    if r['字形补全']:tags.append(r['字形补全'])
    if int(r['字体清理像素'] or 0):tags.append('细直线清理')
    if r['文件'] in hair:tags.append('贯穿划痕')
    if r['文件'] in frag:tags.append('孤立碎片')
    if r['字']=='𠨰':tags.append('残留划痕')
    if r['字'] in ['玄','𤣥']:tags.append('避讳玄')
    r['review_tags']=tags;r['hairline_pixels']=hair.get(r['文件'],0);r['fragment_pixels']=frag.get(r['文件'],0)
    r['sha256']=hashlib.sha256((a.original/r['文件']).read_bytes()).hexdigest();counts.update(tags)
    if tags:review.append(r)
  doc.close();print('GEOMETRY',vol,flush=True)
 dump(ROOT/'sources/manifest.json',rows);dump(ROOT/'sources/page-geometry.json',geometry);dump(ROOT/'reports/review-items.json',review)
 dump(ROOT/'reports/source-audit.json',{'images':len(rows),'categories':dict(collections.Counter(r['类别'] for r in rows)),'review_counts':dict(counts),'review_union':len(review),'pua_characters':[r['Unicode'] for r in rows if r['类别']=='字头' and 0xE000<=ord(r['字'])<=0xF8FF],'size_note':'PNG pixels unchanged; PDF dpi is metadata, frame calibration uses published half-leaf height 193 mm and is an estimate.'})
 with (ROOT/'reports/review-items.csv').open('w',encoding='utf-8-sig',newline='') as fh:
  fields=list(review[0]);w=csv.DictWriter(fh,fieldnames=fields);w.writeheader();w.writerows([{**r,'review_tags':' | '.join(r['review_tags'])} for r in review])
 with zipfile.ZipFile(ROOT/'sources/glyphs-original.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for r in rows:z.write(a.original/r['文件'],r['文件'])
 png=ROOT/'build/png';png.mkdir(exist_ok=True)
 for r in rows:
  dst=png/r['文件'];dst.parent.mkdir(exist_ok=True);shutil.copy2(a.original/r['文件'],dst)
 print(json.dumps({'counts':dict(counts),'union':len(review),'zip_bytes':(ROOT/'sources/glyphs-original.zip').stat().st_size},ensure_ascii=False))
if __name__=='__main__':main()
