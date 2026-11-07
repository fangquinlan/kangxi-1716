# 原始切分算法

此目录保留 Claude 最终工作流的 Python 源码（MIT）。主要阶段：

1. common.py / geom.py：提取 PDF 内嵌扫描，拟合版框、八行栏距和行相位。
2. detect.py / pipeline.py：距离变换区分大字、小字；检测圈号及长细框线。
3. templates.py / feats.py / align.py：参考字体特征与字头序列动态规划对齐。
4. repair.py：补搜；export.py / complete.py / fontclean.py：连通部件取舍和保守二值化。
5. hairline_pass.py / detached_pass.py：划痕和孤立碎片复核式清理。

主要环境变量：KX_SOURCE_ROOT（原PDF目录）、KX_WORK（中间工作目录）、KX_REFERENCE_FONT（用户合法取得的参考字体）、KX_HEADWORDS（字头JSON）、KX_OUT（新导出目录）。源数据和输出路径已参数化；部分历史诊断脚本的标注字体仍使用 Windows 系统字体路径。历史诊断脚本保留阶段特定参数，不能视为统一跨平台入口。

公开的 headwords.json 只保留对齐所需的页内字序、卷名与部首，不重新分发原表附带的现代读音、注音、仓颉等未使用字段。

运行参考顺序：templates.py → pipeline.py 01,02,...,13 → align_all.py → repair.py 按卷 → export.py → 两种后清理。不同阶段还包含人工挑选和排除记录，全部原切分结果和审核标记以 sources/ 中的归档为准。字体可从该归档完整重建，不依赖历史pickle或参考字体。

注意：export.py 会重建 KX_OUT 下的字头目录，应指定新的空导出目录；不要指向唯一原件。原算法 clean_glyph 含≤10像素孔填补，见来源说明。未知许可的参考字体不随本项目发布。
