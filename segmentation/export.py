"""Stage 3: export final 1-bit glyph images and the manifest.

OUT/
  字头PNG/<字>.png                 best occurrence of every headword (one file per character)
  重复字头/<字>_<卷>_<页><面><行>.png  further occurrences of headwords listed more than once
  部首标题/<字>_<卷>_<页><面><行>.png  large radical titles ("X部"), i.e. extra images of the radical character
  字头清单.csv                     one row per exported image
  未匹配字头.csv                    expected headwords without an image
"""
from align import *
from render import clean_glyph, save_1bit
from fontclean import clean_with_font
import csv, shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

OUT = os.environ.get('KX_OUT', os.path.join(ROOT, 'output', '武英殿康熙字典字头'))
VOL_NAMES = {'01': '子集', '02': '丑集', '03': '寅集', '04': '卯集', '05': '辰集', '06': '巳集', '07': '午集',
             '08': '未集', '09': '申集', '10': '酉集', '11': '戌集', '12': '亥集', '13': '補遺備考'}
HEADER = set('康熙字典集上中下補遺備考') | set(BRANCHES)


def loc(it):
    return f"{it['vol']}{VOL_NAMES[it['vol']]}_p{it['pno'] + 1:03d}{'右左'[it['face']]}{it['col'] + 1}"


def collect():
    inst = []
    expected_all = []
    for vol in sorted(VOL_RANGES):
        A = pickle.load(open(os.path.join(WORK, f'align_{vol}.pkl'), 'rb'))
        exp = A['exp']
        juan = jdata()['page_juan']
        got = set()
        for r in A['res']:
            base = dict(vol=vol, pno=r['pno'], ci=r['ci'], face=r['face'], col=r['col'], yc=r['yc'], src='cand')
            if r['match'] >= 0:
                got.add(r['match'])
                inst.append(dict(base, ch=r['ch'], role='match', sim=r['sim'], site_page=r['page'], seq=r['seq'],
                                 juan=juan.get(str(r['page']), ''), j=r['match']))
            elif r.get('title_ch') and r['title_ch'] not in HEADER and r['title_ch'] != '部' and r['title_sim'] >= 0.6:
                inst.append(dict(base, ch=r['title_ch'], role='radical', sim=r['title_sim'], site_page=r['title_page'],
                                 seq=None, juan=juan.get(str(r['title_page']), ''), j=None))
        rp = os.path.join(WORK, f'repair_{vol}.pkl')
        if os.path.exists(rp):
            for k, f in enumerate(pickle.load(open(rp, 'rb'))['found']):
                if f['rejected']:
                    continue
                if f.get('replaces'):
                    pno, ci = f['replaces']
                    inst[:] = [it for it in inst if not (it['vol'] == vol and it['pno'] == pno and it['ci'] == ci and it['src'] == 'cand')]
                    got.discard(f['j'])
                if f['j'] in got:
                    continue
                got.add(f['j'])
                inst.append(dict(vol=vol, pno=f['pno'], ci=k, face=f['face'], col=f['col'], yc=f['yc'], src='repair',
                                 ch=f['ch'], role='match', sim=f['sim'], site_page=f['page'], seq=f['seq'],
                                 juan=juan.get(str(f['page']), ''), j=f['j']))
        # single-slot gaps: an expected entry between two matched neighbours, with exactly one unmatched
        # candidate between them in reading order, is assigned by position (flagged)
        res = A['res']
        pos_of = {r['match']: i for i, r in enumerate(res) if r['match'] >= 0}
        for j in range(len(exp)):
            if j in got or (j - 1) not in pos_of or (j + 1) not in pos_of:
                continue
            i0, i1 = pos_of[j - 1], pos_of[j + 1]
            between = [res[i] for i in range(i0 + 1, i1) if res[i]['match'] < 0 and not res[i]['title']]
            if len(between) == 1:
                r = between[0]
                got.add(j)
                inst.append(dict(vol=vol, pno=r['pno'], ci=r['ci'], face=r['face'], col=r['col'], yc=r['yc'],
                                 src='cand', ch=exp[j][2], role='match', sim=r['best_sim'] if False else 0.0,
                                 site_page=exp[j][0], seq=exp[j][1], juan=juan.get(str(exp[j][0]), ''), j=j,
                                 positional=True))
        for j, (p, s, ch) in enumerate(exp):
            expected_all.append((vol, j, p, s, ch, j in got))
    best = {}
    for k, it in enumerate(inst):
        if it['role'] == 'match' and (it['ch'] not in best or it['sim'] > inst[best[it['ch']]]['sim']):
            best[it['ch']] = k
    for k, it in enumerate(inst):
        if it['role'] == 'match':
            it['role'] = 'primary' if best[it['ch']] == k else 'repeat'
    for it in inst:
        if it['role'] == 'primary':
            it['file'] = f"字头PNG/{it['ch']}.png"
        elif it['role'] == 'repeat':
            it['file'] = f"重复字头/{it['ch']}_{loc(it)}.png"
        else:
            it['file'] = f"部首标题/{it['ch']}_{loc(it)}.png"
    # make names unique (two items at the same column position)
    seen = Counter()
    for it in inst:
        seen[it['file']] += 1
        if seen[it['file']] > 1:
            stem, ext = it['file'].rsplit('.', 1)
            it['file'] = f"{stem}_{seen[it['file']]}.{ext}"
    return inst, expected_all


_SCALE = {}


def vol_scale(vol):
    if vol not in _SCALE:
        _SCALE[vol] = pickle.load(open(os.path.join(WORK, f'repair_{vol}.pkl'), 'rb'))['scale']
    return _SCALE[vol]


def render_page(args):
    from complete import Strips, complete
    vol, pno, items = args
    rec = pickle.load(open(os.path.join(WORK, os.environ.get('KX_CANDS', 'cands'), vol, f'{pno:04d}.pkl'), 'rb'))
    doc = fitz.open(pdf_list()[vol])
    S = Strips(doc, rec)
    reps = None
    out = []
    for it in items:
        if it['src'] == 'cand':
            c = rec['cands'][it['ci']]
        else:
            if reps is None:
                reps = pickle.load(open(os.path.join(WORK, f'repair_{vol}.pkl'), 'rb'))['found']
            c = reps[it['ci']]
        thr = rec['thr']
        info = dict(choice='orig', s=it['sim'])
        try:
            c, info = complete(c, it['ch'], S.get(c['side'], c['k']), vol_scale(vol))
        except Exception as e:  # keep the original mask if completion fails
            info = dict(choice='error:' + repr(e)[:60], s=it['sim'])
        img = clean_glyph(c, thr)
        if img is None:
            out.append((it['file'], None))
            continue
        st = dict(removed=0, cc=None)
        if info['s'] >= 0.7:
            img, st = clean_with_font(img, it['ch'])
        save_1bit(img, os.path.join(OUT, it['file']))
        out.append((it['file'], dict(w=img.shape[1], h=img.shape[0], x0=c['x0'], x1=c['x1'], y0=c['y'],
                                     y1=c['y'] + c['h'], removed=st.get('removed', 0), cc=st.get('cc'),
                                     choice=info['choice'], s=info['s'])))
    return out


def main():
    inst, expected_all = collect()
    if os.path.exists(OUT):
        for d in ('字头PNG', '重复字头', '部首标题'):
            shutil.rmtree(os.path.join(OUT, d), ignore_errors=True)
    for d in ('字头PNG', '重复字头', '部首标题'):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    bypage = defaultdict(list)
    for it in inst:
        bypage[(it['vol'], it['pno'])].append(it)
    jobs = [(v, p, its) for (v, p), its in sorted(bypage.items())]
    info = {}
    t = time.time()
    with ProcessPoolExecutor(10) as ex:
        for out in ex.map(render_page, jobs, chunksize=2):
            for f, d in out:
                info[f] = d
    print('rendered', len(info), f'{time.time() - t:.0f}s', flush=True)
    role_cn = {'primary': '字头', 'repeat': '重复字头', 'radical': '部首标题'}
    rows = []
    for it in sorted(inst, key=lambda i: (i['vol'], i['pno'], i['face'], i['col'], i['yc'])):
        d = info.get(it['file'])
        if d is None:
            continue
        ch = it['ch']
        rows.append([ch, ' '.join(f'U+{ord(x):04X}' for x in ch), it['file'], role_cn[it['role']],
                     it['vol'], VOL_NAMES[it['vol']], it['pno'] + 1, '右' if it['face'] == 0 else '左', it['col'] + 1,
                     d['x0'], d['y0'], d['x1'], d['y1'], it['site_page'] or '', it['seq'] or '', it['juan'],
                     round(d['s'], 3), '补搜' if it['src'] == 'repair' else ('位置推定' if it.get('positional') else ''), d['removed'],
                     {'orig': '', 'union': '补全', 'font': '重组'}.get(d['choice'], d['choice'])])
    with open(os.path.join(OUT, '字头清单.csv'), 'w', encoding='utf-8-sig', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['字', 'Unicode', '文件', '类别', '册', '集', 'PDF页', '面', '行(自右)', '页面x0', '页面y0', '页面x1',
                    '页面y1', '网站页码', '页内序号', '网站卷', '字形相似度', '来源', '字体清理像素', '字形补全'])
        w.writerows(rows)
    missing = [e for e in expected_all if not e[5]]
    with open(os.path.join(OUT, '未匹配字头.csv'), 'w', encoding='utf-8-sig', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['字', 'Unicode', '册', '网站页码', '页内序号'])
        for vol, j, p, s, ch, _ in missing:
            w.writerow([ch, f'U+{ord(ch):04X}', vol, p, s])
    json.dump(inst, open(os.path.join(WORK, 'export_instances.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    roles = Counter(it['role'] for it in inst if info.get(it['file']))
    stats = dict(expected_entries=len(expected_all), matched_entries=len(expected_all) - len(missing),
                 missing_entries=len(missing), unique_characters=roles['primary'], repeat_images=roles['repeat'],
                 radical_title_images=roles['radical'],
                 font_cleaned_images=sum(1 for d in info.values() if d and d['removed'] > 0),
                 completed_images=sum(1 for d in info.values() if d and d['choice'] in ('union', 'font')))
    json.dump(stats, open(os.path.join(OUT, '统计.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(stats)


if __name__ == '__main__':
    main()
