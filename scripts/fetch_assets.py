"""Fetch checksum-verified release assets for previews and reproducible builds."""
import argparse,gzip,hashlib,json,shutil,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--all',action='store_true',help='also fetch SVG and per-glyph measurements');a=p.parse_args()
 for item in json.loads((ROOT/'assets.json').read_text(encoding='utf-8')):
  if item.get('optional') and not a.all:continue
  dst=ROOT/item['path'];dst.parent.mkdir(parents=True,exist_ok=True)
  if not dst.exists() or hashlib.sha256(dst.read_bytes()).hexdigest()!=item['sha256']:
   tmp=dst.with_suffix(dst.suffix+'.part');print('DOWNLOAD',item['name'],flush=True)
   with urllib.request.urlopen(item['url'],timeout=120) as r,tmp.open('wb') as w:shutil.copyfileobj(r,w)
   if hashlib.sha256(tmp.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Checksum mismatch: '+item['name'])
   tmp.replace(dst)
  if item.get('unzip_to'):
   target=(ROOT/item['unzip_to']).resolve()
   with zipfile.ZipFile(dst) as z:
    for n in z.namelist():
     if not (target/n).resolve().is_relative_to(target):raise ValueError('Unsafe archive path')
    z.extractall(target)
  if item.get('gunzip_to'):
   target=ROOT/item['gunzip_to'];target.parent.mkdir(parents=True,exist_ok=True)
   with gzip.open(dst,'rb') as r,target.open('wb') as w:shutil.copyfileobj(r,w)
  print('OK',item['name'],flush=True)
if __name__=='__main__':main()
