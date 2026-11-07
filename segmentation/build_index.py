from common import ROOT, WORK, os
"""Browsing index: one HTML page per volume (scan glyph beside a modern-font rendering) + global search page."""
import csv, json, os, sys, html
from collections import defaultdict
from fontTools import subset
from fontTools.ttLib import TTFont
OUT = os.environ.get('KX_OUT', os.environ.get('KX_OUT', os.path.join(ROOT, 'output', '武英殿康熙字典字头')))
FONTDIR = 'C:\\Users\\Administrator\\AppData\\Local\\Microsoft\\Windows\\Fonts'
FONTS = [('KXSrc', 'SourceHanSansCN-Regular-NoHangul.otf', '思源黑体'), ('KXPl1', 'PlangothicP1-Regular.otf', '遍黑 P1'), ('KXPl2', 'PlangothicP2-Regular.otf', '遍黑 P2')]
VOL_NAMES = {'01': '子集', '02': '丑集', '03': '寅集', '04': '卯集', '05': '辰集', '06': '巳集', '07': '午集', '08': '未集', '09': '申集', '10': '酉集', '11': '戌集', '12': '亥集', '13': '補遺備考'}
CSS = '*{box-sizing:border-box}body{margin:0;background:#f4f3ef;color:#222;font:14px system-ui,sans-serif}\nheader{position:sticky;top:0;background:#fffdf8;padding:12px 20px;border-bottom:1px solid #ccc;z-index:2}\nh1{font-size:20px;margin:0 0 6px}a{color:#3c5a78}nav a{margin-right:10px}\nmain{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px;padding:16px}\n.c{background:#fff;border:1px solid #ddd;border-radius:5px;padding:8px}\n.p{display:grid;grid-template-columns:1fr 1fr;gap:6px;align-items:center}\n.p img{width:100%;aspect-ratio:1;object-fit:contain}\n.f{font-size:72px;line-height:1;text-align:center;font-family:KXSrc,KXPl1,KXPl2,serif}\n.m{font-size:12px;color:#555;line-height:1.5;margin-top:4px}.m b{font-size:15px;color:#000}\n.low{border:2px solid #c88a2b}.rep{border:2px solid #4a8bc4}\ninput{font:inherit;padding:6px 10px;width:320px;max-width:70vw}\n'

def font_css(prefix):
    return ''.join(("@font-face{font-family:%s;src:url('%sfonts/%s.woff2') format('woff2');font-display:swap}" % (n, prefix, n) for n, _, _ in FONTS))

def make_fonts(chars):
    os.makedirs(os.path.join(OUT, 'fonts'), exist_ok=True)
    remaining = set(chars)
    lic = ['# 对照字体', '每字优先用思源黑体，未收录的依次用遍黑 P1、P2；只含本书用字的子集，仅供对照，不替换原扫描字形。', '']
    for name, fn, label in FONTS:
        path = os.path.join(FONTDIR, fn)
        cmap = TTFont(path, lazy=True).getBestCmap()
        use = sorted((c for c in remaining if ord(c) in cmap))
        remaining -= set(use)
        opts = subset.Options()
        opts.flavor = 'woff2'
        opts.layout_features = []
        opts.name_IDs = ['*']
        opts.notdef_outline = True
        f = subset.load_font(path, opts)
        sub = subset.Subsetter(opts)
        sub.populate(unicodes=[ord(c) for c in use] or [32])
        sub.subset(f)
        subset.save_font(f, os.path.join(OUT, 'fonts', name + '.woff2'), opts)
        lic.append(f'## {label}: {fn}，覆盖 {len(use)} 字，许可 SIL Open Font License 1.1')
        print(label, len(use))
    lic.append(f'\n三套字体均未收录：{len(remaining)} 字')
    open(os.path.join(OUT, 'fonts', '字体说明.txt'), 'w', encoding='utf-8').write('\n'.join(lic))

def card(r):
    cls = 'c'
    if r['类别'] == '重复字头' or r['类别'] == '部首标题':
        cls += ' rep'
    elif r['字形相似度'] and float(r['字形相似度']) < 0.8:
        cls += ' low'
    flags = []
    if r['来源']:
        flags.append(r['来源'])
    if r['字体清理像素'] not in ('', '0'):
        flags.append('清理%s' % r['字体清理像素'])
    if r['字形补全']:
        flags.append(r['字形补全'])
    if r['类别'] != '字头':
        flags.append(r['类别'])
    ch = html.escape(r['字'])
    src = html.escape(r['文件'])
    return f'''<div class="{cls}" id="{ch}"><div class="p"><img loading="lazy" src="{src}" alt="{ch}"><div class="f">{ch}</div></div><div class="m"><b>{ch}</b> {r['Unicode']}<br>PDF {r['PDF页']} {r['面']}{r['行(自右)']}行 · 网站页 {r['网站页码']}<br>相似度 {r['字形相似度']} {' · '.join(flags)}</div></div>'''

def main():
    rows = list(csv.DictReader(open(os.path.join(OUT, '字头清单.csv'), encoding='utf-8-sig')))
    stats = json.load(open(os.path.join(OUT, '统计.json'), encoding='utf-8'))
    make_fonts(set((r['字'] for r in rows)))
    byvol = defaultdict(list)
    for r in rows:
        byvol[r['册']].append(r)
    nav = ' '.join((f'<a href="{v}_{VOL_NAMES[v]}.html">{v} {VOL_NAMES[v]}</a>' for v in sorted(byvol)))
    for v, rs in sorted(byvol.items()):
        body = '\n'.join((card(r) for r in rs))
        page = f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>{v} {VOL_NAMES[v]} · 武英殿康熙字典字头</title><style>{font_css('')}{CSS}</style><header><h1>{v} {VOL_NAMES[v]}（{len(rs)} 图）</h1><nav><a href="index.html">总览/搜索</a>{nav}</nav></header><main>{body}</main></html>"""
        open(os.path.join(OUT, f'{v}_{VOL_NAMES[v]}.html'), 'w', encoding='utf-8').write(page)
    data = [[r['字'], r['文件'], r['册'], r['PDF页'], r['面'] + r['行(自右)'], r['字形相似度'], r['类别']] for r in rows]
    js = 'const D=' + json.dumps(data, ensure_ascii=False) + ';function go(){const q=document.getElementById(\'q\').value.trim();const m=document.querySelector(\'main\');if(!q){m.innerHTML=\'\';return}const s=new Set([...q]);const hits=D.filter(d=>s.has(d[0])).slice(0,400);m.innerHTML=hits.map(d=>`<div class="c"><div class="p"><img src="${d[1]}"><div class="f">${d[0]}</div></div><div class="m"><b>${d[0]}</b> ${d[6]} · 册${d[2]} PDF ${d[3]} ${d[4]}行 · 相似度 ${d[5]}</div></div>`).join(\'\')}'
    s = '<br>'.join((f'{k}: {v}' for k, v in stats.items()))
    page = f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>武英殿康熙字典字头</title><style>{font_css('')}{CSS}</style><header><h1>武英殿康熙字典 · 字头字图</h1><nav>{nav}</nav><p><input id="q" placeholder="输入要查的字（可多个）" oninput="go()"></p></header><div style="padding:12px 20px" class="m">{s}</div><main></main><script>{js}</script></html>"""
    open(os.path.join(OUT, 'index.html'), 'w', encoding='utf-8').write(page)
    print('pages', len(byvol))
if __name__ == '__main__':
    main()
