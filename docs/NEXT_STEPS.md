# 后续工作

按"解锁条件"排序：前面的是被外部条件卡住的，后面的是自己就能做的。

---

## 1. 解锁 live-model 实验（被 API key 阻塞）

### 1.1 先确认模型 id，不要猜

```bash
cd D:\python\PyCharm\Project\06_evomemory
# 1) 在 .env 里写好 DEEPSEEK_API_KEY，模型名先留空
# 2) 列出 provider 实际支持的模型
PYTHONPATH=src python -c "
from evomemory.config import Settings
from evomemory.llm import DeepSeekClient
s = Settings.from_env(); s.require_api()
print(DeepSeekClient(s.api_key, 'placeholder', s.base_url).list_models())
"
```

把返回列表中**确实存在**的 Flash 型号写进 `DEEPSEEK_MODEL`。如果列表里没有 Flash 型号，就把错误信息记在这里，**不要**擅自换成更贵的推理模型。

> 为什么不让代码猜：项目里 `Settings.require_api()` 在 `model` 为空时直接抛异常，`DeepSeekClient` 在构造时也拒绝空 model。这条约束对应决策 D12 之前的第 3.2 条。

### 1.2 冒烟测试（先花 1-2 次调用确认链路）

```bash
PYTHONPATH=src python -m evomemory.runner --modes A --llm api --per-family 1
```

检查 `artifacts/run_*/mode_A/memory.db` 里的 `llm_calls` 表：`model` 字段是否是预期型号、`input_tokens/output_tokens` 是否是 provider 回报（不是估算）、`ok=1`。

### 1.3 正式实验（预算估算）

```bash
PYTHONPATH=src python -m evomemory.runner --modes ABCD --llm api --per-family 6 \
  --max-calls 150 --max-tokens 400000
```

调用量估算：每个任务约 6-8 次 action 调用 + 1 次 extract（+D 模式 1 次 critique/reconstruct/条）。12 个任务 × 4 模式 ≈ **100-150 次**，在 200 次预算内但接近上限。建议先跑 A 与 D 两个模式（约 50-70 次），确认效应方向后再决定是否补 B/C。

### 1.4 记录什么

运行器会自动写：`results.json`（含 `evidence_class=live_model`）、每模式的 `memory.db`、`llm_calls` 全量台账。**不要手工修改结果文件**；若要补充解读，写进 `docs/EXPERIMENT_RESULTS.md` 新增的一节，并注明 `evidence_class=live_model`。

### 1.5 必须先算功效

12 个 cell 的 MDE 是 ±57pp。如果 live 实验的 A/D 差异小于这个数，**结论只能是"未检出"**，不能写成"记忆无效"也不能写成"略有提升"。

---

## 2. 视觉模型接入（被 GPU / 预算阻塞）

现状：`PixelFactsPerceiver` 只产出可测事实（尺寸、主色、色数），并显式标记 `is_semantic=False`。这在语义上是诚实的，但它**不能**回答"这张截图里用户在问什么"。

接入路径：

1. 实现 `Perceiver` 协议（`describe(bytes) -> VisualDescription`），`is_semantic=True`；
2. 在 `MultimodalMemory(perceiver=...)` 注入；
3. `Harness._caption_map()` 会自动把 caption 拼进检索文档，无需改其他代码；
4. 补一条测试：同一 experience 在有/无 caption 时的检索排名差异。

**验收标准**：`C` 模式与 `B` 模式的 per-task 结果出现差异，且差异方向可解释。在此之前，任何"多模态更好"的说法都不成立。

可用于离线验证的低成本方案：用 `SidecarCaptionPerceiver` + 人工写的 caption 构造一批带标注的图像证据，验证**跨模态检索通路**（不验证视觉模型质量）。

---

## 3. 扩大任务集（自己就能做）

当前 12 个 cell → MDE ±57pp。目标 50-100 个 cell → MDE 约 ±20pp 到 ±15pp。

做法：`bench.build_dataset(seed, per_family=N)` 直接加大 `per_family`。需要同时确认：

- 每个 family 的失败上限仍是 1 次，所以成功率上限仍是 `(n - 2)/n`；任务越多，上限越接近 100%，指标越有意义；
- 新增 family（例如数值阈值型规则、组合条件型规则）以测试"规则形状"的泛化，而不只是"规则内容"。

**建议新增的 family**：
- `triage`：规则依赖优先级之间的相对关系而非绝对阈值；
- `merge`：规则依赖两个字段的联合条件（tier=gold **且** region=eu 才排除），用来测 critique 的适用性判断在条件组合下的表现。

---

## 4. 把统计 gate 接进批次级演化

现状：单条经验的准入是**程序化检查**（9 项），没有统计显著性；`metrics.mde_binary` 只是报告用。

下一步：当"修订某条经验"（`ExperienceStore.revise`）时，用配对 bootstrap 判断新旧版本在 held-out 任务上的差异是否显著，不显著则不 supersede。这正是 Recuris 的 `held_out_paired_gate`，但要用**自己的**实现（其源码 Apache-2.0，可参考不可复制其 driver）。

注意审计里的教训：该 gate 在 item 级重采样时，如果所有 item 差异相同会给出零宽度 CI，过度自信。实现时必须显式处理零方差。

---

## 5. 跨会话长期记忆

现状：一次 run 一个 `.db`，A/B/C/D 之间隔离是刻意的。

需要一层"持久记忆库"：`data/evomemory.db` 作为跨 run 的经验池，run 级数据库只记录本次执行。设计要点：

- 经验需要带 `scope`（是否跨项目可用）；
- 跨 run 复用会让 `negative_transfer_rate` 成为更重要的指标；
- 必须有过期机制：一条经验在 N 次检索后若从未被采纳（`critique=reject` 或注入后仍失败），应降级。

---

## 6. 记忆压缩对**生成侧**的影响（CMA 没做的实验）

现状：`fidelity()` 已经能测量缩略图的内容几何与色彩偏差（见 `artifacts/multimodal/REPORT.md`）。

下一步：把"检索用压缩表示、生成用原图"扩展成三种配置——(a) 检索压缩 / 生成原图（CMA 的做法）；(b) 检索与生成都用缩略图；(c) 检索与生成都用 caption。测 (b)(c) 相对 (a) 的任务成功率下降曲线。

这是本项目相对 CMA 最有区分度的问题，因为它的代码里压缩只作用于检索侧，从未回答"如果生成也只能看压缩表示会怎样"。

---

## 7. 负迁移压力测试（自己就能做）

现状：`negative_transfer_rate` 只在自然发生的跨 family 注入时才能观测到（B 模式 25%）。

主动版本：故意向记忆库注入三类有害经验，看 gate 与 critique 能否拦住：

1. **过期经验**：规则曾经是对的，但环境已变（例如 gold 规则被反转为 bronze）；
2. **近似但错误**：lesson 文本与正确规则高度相似，只有一处数值不同；
3. **正确但越界**：规则本身正确，但 applicability 条件写得太宽。

第 2 类是对 `HashEmbedder` 去重阈值的直接考验——语义改写的近似重复不会被词法余弦捕获，这会是当前实现的一个已知漏点。

---

## 8. 已知缺口（不打算在近期修）

| 缺口 | 影响 | 若要修 |
|---|---|---|
| `HashEmbedder` 是词法不是语义 | 同义改写召不回 | 实现 `Embedder` 协议，接真实 embedding |
| 精确线性检索 | >10⁴ 条经验后变慢 | 换 ANN，但会牺牲可复算性 |
| 无并发、无分布式 | 单进程串行 | 需要先补 `active()` 的部分唯一索引 |
| `active()` 的"每 id 一个活动版本"靠代码保证 | 并发写入时可能违反 | 加 `CREATE UNIQUE INDEX ... WHERE status='active'`（SQLite 支持部分索引） |
| family 只有两个且规则形状相似 | 结论不能外推到其他规则形状 | 见第 3 节 |
