# 实验结果

**证据等级：`harness_selfcheck`（离线脚本策略）。**
本文所有数字都由 `scripts/` 与 `src/evomemory/runner.py` 生成，可重放：

```bash
PYTHONPATH=src python3 -m evomemory.runner --modes ABCD --llm scripted --per-family 6 --out artifacts
```

原始记录：`artifacts/run_<timestamp>/results.json`、每个模式的 `memory.db`、`summary.md`。

> **这些数字不能用来声称"记忆提升了大模型的任务成功率"。** 策略是确定性规则跟随器，不是模型。它证明的是管线闭合、门控生效、指标可计量。真实模型上的效应量见 `NEXT_STEPS.md` 的待办实验。

---

## 1. 主结果

任务数 12（2 个 family × 6），MDE（本样本量下可检出的最小效应）≈ **±57 pp**。

| 模式 | 成功率 | retrieval hit | 负迁移率 | token/任务 | 工具调用/任务 | 写入经验 | 准入经验 |
|---|---|---|---|---|---|---|---|
| A 无记忆 | **0.0%** (0/12) | — | 0.0% | 4232.0 | 7.50 | 0 | 0 |
| B 简单向量记忆 | 58.3% (7/12) | 100% (10/10) | **25.0%** | 3813.1 | 6.08 | 12 | 12 |
| C 多模态记忆 | 58.3% (7/12) | 100% (10/10) | **25.0%** | 3813.1 | 6.08 | 12 | 12 |
| D 完整生命周期 | **83.3%** (10/12) | 100% (10/10) | **0.0%** | 3399.1 | 5.83 | 12 | **4** |

相对 A 的差值：B/C 成功率 +58.3pp、token −419；D 成功率 +83.3pp、token −833、负迁移 0。

**C 与 B 完全相同**（逐位相同，`test_determinism.py` 保证）。这是预期结果：离线环境没有图片，caption 通道为空，多模态模块在这组任务上不产生任何信号。任何"离线 C > B"的结论都应当是 bug。

---

## 2. 逐任务观察（这是最有信息量的部分）

### 2.1 模式 D：每个 family 只失败一次

```
escalation_00  FAIL   inj=0    ← 第一次遇到这个 family，隐藏规则未知
escalation_01  OK     inj=1
escalation_02  OK     inj=2
escalation_03  OK     inj=2
escalation_04  OK     inj=2
escalation_05  OK     inj=2
closure_00     FAIL   inj=0    ← 第一次遇到 closure
closure_01..05 OK     inj=1..2
```

成功率 83.3% **等于**该任务集的结构性上限 `(12 − 2) / 12`。也就是说：D 模式在学会一个 family 的规则之后，没有再犯过任何一次错。

### 2.2 模式 B：记忆会污染自己

```
escalation_00  FAIL   ← 无记忆
escalation_01  OK
escalation_02  OK
escalation_03  OK
escalation_04  OK
escalation_05  FAIL   inj=3    ← 学完之后又退化了
closure_00     FAIL   inj=3    ← 注入了 escalation 的经验，负迁移
closure_01     FAIL   inj=3    ← 又注入了错的经验
closure_02..05 OK
```

这是本次实验最值得记录的一条：**没有校验和去重的记忆系统会随时间变差。** B 模式把每条成功轨迹都变成一条经验（12 条全部 active），其中大量是"这次成功了，照这个顺序做"的同义重复。当同 family 的 active 经验超过 top-k，检索回来的三条里可能只剩一条真正的规则经验、另外两条是无用的成功叙事；而 `escalation_05` 就是这样重新踩回了 gold 记录。

D 模式把 12 个候选里的 8 个拒掉了（`not_duplicate` 等检查），只留 4 条 active，检索集始终很小且有效。

**这条发现直接对应审计里的一个观察**：MemHarness 的去重在其发布脚本里被显式关闭（`train_alfworld.sh:135-136`），而它的 README 把语义去重列为特性。本项目的实验给出了关闭去重的一个具体代价。

### 2.3 负迁移是可观测的

B/C 的 25% 负迁移来自 `closure_00` 与 `closure_01`：这两次注入的经验来自 `escalation` family，于是策略按 escalation 的规则（note 含 `hold` 不动）处理 closure 任务，而 closure 真正的排除条件是 note 含 `audit`——结果动了本该保留的记录。

D 模式为 0，因为 `MemoryCritic` 在 family 不匹配时直接 `REJECT`。

---

## 3. 校验门的实际作用

| 指标 | B/C | D |
|---|---|---|
| 候选经验数 | 12 | 12 |
| 通过准入 | 12 | 4 |
| 其中被 `not_duplicate` 拒绝 | — | 8 |
| active 经验总数 | 12 | 4 |
| 证据可解析率 | 100% | 100% |

D 的 8 次拒绝全部来自重复检查：同一个 family 的成功经验反复被提取成几乎相同的 lesson。校验门在这里的角色不是"提高单条质量"，而是**控制记忆库的规模与多样性**。

一个诚实的补充：D 之所以能检出重复，是因为 `ExperienceValidator` 用了 `HashEmbedder` 的余弦相似度（阈值 0.95）。如果是语义改写而不是逐字重复，这个阈值会漏。这是 `HashEmbedder` 的真实局限，不是设计意图。

---

## 4. 成本

| 模式 | token/任务 | 相对 A | 工具调用/任务 |
|---|---|---|---|
| A | 4232 | — | 7.50 |
| B/C | 3813 | −10% | 6.08 |
| D | 3399 | −20% | 5.83 |

记忆在这里**降低了**成本，因为失败的任务会反复重试（A 模式每步都要列表、逐条写、最后仍然失败）。这不是"记忆省钱"的普遍结论——它只说明在**有确定性正确答案、且不知道规则就会重复犯错**的任务上，一次成功的学习比反复试错便宜。

token 数是估算（4 字符/token），只能做模式间相对比较。

---

## 5. 复现性与正确性

| 检查 | 结果 |
|---|---|
| 两次相同配置的运行是否逐位相同 | **是**（`test_determinism.py::test_two_runs_are_identical`） |
| B 与 C 在无图像环境下是否完全相同 | **是** |
| 一个什么都不做的 agent 是否必然失败 | **是**（`test_do_nothing_agent_fails`） |
| 一个正确执行的 agent 是否必然成功 | **是**（`test_correct_agent_succeeds`） |
| 每个任务是否都有目标和排除项 | **是**（`test_every_task_has_targets_and_exclusions`） |
| Trace 能否被重放并复现同样的结局 | **是**（`test_replay_reproduces_the_recorded_outcome`） |
| 中断后能否跳过已结算任务继续 | **是**（`test_recovery_resume_runs_only_the_remainder`） |

开发过程中确实修掉过一个真的破坏复现性的 bug：检索排序的并列项原本按 `experience_id`（uuid4）决胜负，导致**两次完全相同的运行召回顺序不同**，表现为 B=50%、C=58%（而这两者本应完全相同）。修正方式是改为按文档文本 / 稳定插入序决胜负。这条记录在 `docs/DECISIONS.md` 的 D11。

---

## 6. 多模态压缩研究（可独立测量）

`scripts/compression_study.py`，4 张合成图 × 4 种尺寸 × 3 种缩放方式 = 48 个测量单元。
原始数据：`artifacts/multimodal/compression_study.json`，报告：`artifacts/multimodal/REPORT.md`。

三种表示的 token 代价（**估算**，16px patch / 2×2 merge）：

| 表示 | 构成 |
|---|---|
| 纯文本摘要 | caption 的 token |
| 摘要 + 448 缩略图 | caption + 196（估算） |
| 摘要 + 原图 | caption + 原生分辨率 token |

保真度是**实测**的，不依赖任何模型：比较原图与派生图的内容区域宽高比、平均色距、主色通道是否保持。

关键实测结论：

1. **`squash`（直接拉成正方形）在非方形图上必然破坏几何**：300×900 的图内容宽高比误差达到 2.0（即 3 倍失真）。这正是 CMA-Harness 的做法，而它的代码里没有任何关于这个损失的测量。
2. **`letterbox` 保住几何**（宽高比误差 < 0.02），但代价是**全局色彩统计被黑边稀释**：180×140 的小图缩到 896 画布时，平均色距达到 245.8。也就是说"不失真"和"保颜色"在缩略图上不能同时做到。
3. **`crop` 保住颜色、牺牲视野**：非方形输入的内容宽高比误差 0.22。

这三条是可复现的测量，不是主张。它们支撑下一阶段一个具体的研究问题：**当记忆压缩同时作用于"检索"与"生成"两侧时，保真度下降如何传递到任务表现**——CMA 只压缩了检索侧，没有回答这个问题。
