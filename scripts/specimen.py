from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1]
font_path=str(R/'fonts/OFLKangxi-Regular.otf')
img=Image.new('RGB',(1500,1140),'#f3f0e7');d=ImageDraw.Draw(img);label=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',28);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
d.text((55,35),'开放康熙字典体 / OFL Kangxi',font=label,fill='#322f25')
d.text((55,80),'1716 内府刊本 · 38 字框修订 · 保留避讳缺笔与部分划痕',font=small,fill='#70694f')
text='天地玄黃宇宙洪荒日月盈昃辰宿列張寒來暑往秋收冬藏閏餘成歲律呂調陽雲騰致雨露結爲霜金生麗水玉出崑岡'
f=ImageFont.truetype(font_path,48)
for j in range(3):
 for i,ch in enumerate(text[j*24:(j+1)*24]):d.text((60+i*55,145+j*76),ch,font=f,fill='#151815')
d.line((55,390,1445,390),fill='#c9c0aa',width=2)
for j in range(7):
 for i,ch in enumerate(text[j*8:(j+1)*8]):d.text((680-j*83,440+i*66),ch,font=f,fill='#151815')
big=ImageFont.truetype(font_path,85)
for j,row in enumerate(['一丨小口田','國鬱龜龍麤','玄𤣥𠈿𠯉㸮','𢆟炢鑿𦤈栴']):
 for i,ch in enumerate(row):d.text((840+i*108,430+j*135),ch,font=big,fill='#151815')
d.text((840,1010),'部分字形仍残有划痕笔画，未做进一步清理。',font=small,fill='#70694f')
img.save(R/'proof/specimen.png');print('specimen.png')
