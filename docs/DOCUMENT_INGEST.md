# PDF、图片和 Markdown 知识导入

## 能力与限制

- 原生文本 PDF 按页提取文本；有图片或没有文本层的页面渲染后使用本地 Tesseract OCR。图文混排页分别保留文本层和 OCR 中新增的文字，去掉与文本层完全相同的 OCR 行。
- PNG、JPEG、TIFF、BMP、WebP 使用 OCR；Markdown 支持本地内联图片、引用式图片和 `<img src>`。代码块中的示例链接不会被当作附件。远程图片不会自动下载，会给出提示。
- Markdown 标题和代码围栏独立处理；超长普通段落按长度拆分，代码围栏和表格优先保持完整。过大的代码块或表格可能超过目标块大小。
- 每个块记录来源类型；PDF 结果标注页码，独立图片标注 OCR，Markdown 引用图片保留原文件行号和图片路径。图片文件变化会触发引用它的 Markdown 重新导入。
- OCR 识别出的**文本**可进入 BM25 和文本 embedding 索引。这不是图片 embedding、跨模态检索或图表视觉理解。页内双栏阅读顺序、复杂表格结构、图形关系、小字、旋转和手写的识别质量尚未充分评测；OCR 文本可能有误，回答时应回看原件。
- OCR 缺失或识别错误不能当成原文真实内容：无法运行时导入报告错误，更新失败时保留旧版本；图像页没有识别到文字会提示。加密、损坏的 PDF 需另行处理。

## 安装与运行

Python 3.10+，安装项目可选依赖：

```powershell
python -m pip install -e ".[dev,rag,documents]"
```

OCR 需要本机 Tesseract 程序及 `chi_sim` 和 `eng` 语言数据。安装程序后将其加入 PATH，或设置 `EVOMEMORY_TESSERACT` 指向可执行文件。默认从项目 `cache/tessdata` 读取语言数据，若该目录不存在则使用 Tesseract 系统默认数据目录；也可设置 `EVOMEMORY_TESSDATA` 和 `EVOMEMORY_OCR_LANG`。例如系统已有英文数据时：

```powershell
New-Item -ItemType Directory -Force cache/tessdata | Out-Null
Copy-Item "$env:ProgramFiles/Tesseract-OCR/tessdata/eng.traineddata" cache/tessdata/
Invoke-WebRequest https://github.com/tesseract-ocr/tessdata_fast/raw/main/chi_sim.traineddata -OutFile cache/tessdata/chi_sim.traineddata
$env:PYTHONPATH = "src"
python -m evomemory.kb_cli import --corpus data/kb_corpus --embedder fastembed
python -m evomemory.kb_cli search --query "故障代号" --mode hybrid --embedder fastembed
```

可将自己的 `.pdf`、`.md`、`.txt` 和图片放到指定的 `--corpus` 文件夹；不要把敏感文档放进示例语料或提交到 Git。引入本地模型和语言文件不会消耗模型 API 次数，首次下载需要网络。默认模型缓存在项目的 `cache/huggingface/`，Windows 与 Linux 分目录，避免跨系统符号链接不可用。

## 实际验证（构造文件）

构造一份包含英文原生文字与中文告警截图的双页 PDF（第二页为扫描页）、同一张独立 PNG 和引用该图的 Markdown。在 Windows 本机使用中文/英文 Tesseract、PDFium 与 `BAAI/bge-small-zh-v1.5`：

- OCR 实际识别 `ALPHA-731` 和“立即回滚”等文字。
- BM25 与真实 512 维文本 embedding 均可检索 OCR 结果，来源分别指向 PDF 第 1、2 页、独立 PNG 和 Markdown 引用行。
- 原生 PDF 文本 `BRAVO-942` 可定位到第 1 页；同页重复的原生文本 OCR 行不再单独重复索引。
- 这是一组构造文件的功能验证，不是 OCR 准确率、检索质量或复杂图表理解的统计结论。

原始结果：`artifacts/document-verification/real-c734b90a0ecc4aa69a5acc465e611d95/results.json`。
