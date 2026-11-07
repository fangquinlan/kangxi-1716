from common import ROOT, WORK, os
"""Write 说明.md into the output folder from 统计.json and the manifest."""
import csv, json, os
from collections import Counter
OUT = os.environ.get('KX_OUT', os.environ.get('KX_OUT', os.path.join(ROOT, 'output', '武英殿康熙字典字头')))
st = json.load(open(os.path.join(OUT, '统计.json'), encoding='utf-8'))
rows = list(csv.DictReader(open(os.path.join(OUT, '字头清单.csv'), encoding='utf-8-sig')))
prim = [r for r in rows if r['类别'] == '字头']
low = sum((1 for r in prim if r['字形相似度'] and float(r['字形相似度']) < 0.6))
src = Counter((r['来源'] for r in rows if r['来源']))
comp = Counter((r['字形补全'] for r in rows if r['字形补全']))
byvol = Counter((r['集'] for r in rows if r['类别'] != '部首标题'))
text = f"# 武英殿康熙字典 · 字头字图\n\n来源：`D:\\kangxi\\康熙字典.01–13.*.pdf`（子集至亥集十二册 + 補遺備考），字头序列以 `康熙字典_页码字头.json` 为准。\n\n## 数量\n\n- 网站字头条目 {st['expected_entries']} 条，已全部配图 {st['matched_entries']} 条（缺 {st['missing_entries']}）。\n- `字头PNG/`：{st['unique_characters']} 个文件，每个字一张（同字多次出现时取与字体最相符的一张）。\n- `重复字头/`：{st['repeat_images']} 张，网站本身重复收录的字（如部首字在部首页和补遗中各出现一次）的其余出处。\n- `部首标题/`：{st['radical_title_images']} 张，正文中“某部”标题里的大号部首字（不是字头，单独存放，供挑选）。\n- 各册图数：{'、'.join((f'{k} {v}' for k, v in byvol.items()))}。\n\n## 文件\n\n- 图片：1 位黑白 PNG，纯黑字、纯白底，四周各留 10 像素白边，保持原书比例，不缩放。\n- `字头清单.csv`：每张图一行，含 Unicode、所在 PDF 页/面/行、页面坐标、网站页码与页内序号、与参考字体的字形相似度、来源标记。\n- `index.html` 与各册 `NN_X集.html`：浏览与搜索，每张字图旁附思源黑体/遍黑字形对照（`fonts/` 为本书用字子集）。\n- `未匹配字头.csv`：没有配上图的条目（本次为空）。\n\n## 需要留意的条目\n\n- 字形相似度 < 0.6 的字头 {low} 个，多为原书异体（与现代字体字形不同）或印刷残损；索引页中以橙框标出，建议抽看。\n- 来源标记：补搜 {src.get('补搜', 0)} 个（按字体模板在原位附近重新定位、重新组字）；位置推定 {src.get('位置推定', 0)} 个（参考字体缺字或字形差异过大，按前后字头之间唯一的大字推定，请核对）。\n- 字形补全：{comp.get('补全', 0)} 张补回了未连上的部件（如细笔画的“立”“亠”头、分离的偏旁），{comp.get('重组', 0)} 张按字体轮廓重组以去掉粘连杂物；字体清理去除细直杂线 {st.get('font_cleaned_images', 0)} 张；去除贯穿字形的细划痕 {st.get('hairline_cleaned_images', 0)} 张、字外孤立的框角/划痕碎片 {st.get('fragment_cleaned_images', 0)} 张（清单“划痕碎片清理像素”列）。\n- 少数字头原书墨色斑驳（版面磨损），二值化后笔画内有细小白点，这是原书实况，未作填补。\n\n## 方法概要\n\n1. 每面先找外框（上下边框与书口侧竖框），按框线做轻微倾斜校正；用界行竖线和小字双行之间的空白拟合 8 行的行距与相位，缺界行的版面再用大字位置校正半行相位。\n2. 在每一行内按笔画粗细（距离变换）找字头：字头笔画明显粗于双行小字；带圈的“增”等圈号用圆环模板检出后整体去掉；界行、书名框的竖边按“细而长的竖线”去掉，粗笔画受保护。\n3. 用参考字体渲染网站给出的字头序列，与检出的大字做动态规划对齐（容许跳过、容许“某部”标题与卷首题名），得到每张图对应的字；对齐弱或缺漏处再用字体模板在前后字头之间定位重组。\n4. 已知是哪个字后，用字体轮廓补全未连上的部件、剔除粘连的书名框边线等细直杂线；最后对原扫描灰度做滞后阈值二值化（只向已确认的笔画生长，不向已判定为框线的像素生长）。\n\n参考字体只用于定位、识别和取舍连通块，不生成、不替换任何笔画；图中笔画全部来自原扫描。\n"
open(os.path.join(OUT, '说明.md'), 'w', encoding='utf-8').write(text)
print(text)
