# 实验结果

**本文包含两组不同证据等级的实验：**

| 组 | 证据等级 | 位置 | 能说明什么 |
|---|---|---|---|
| 离线 A/B/C/D | `harness_selfcheck` | §1-§6 | 管线是否闭合、门控是否生效、指标是否可计量 |
| 真实模型 A/D | `live_model` | **§7** | 真实 reasoning 模型下管线能否跑通，以及在真实模型上暴露出的两个缺陷 |

两组**都不能**用来宣称"记忆提升了大模型的任务成功率"。第二组的 6 个任务远低于可检出效应所需样本量（见 §7.4）。

---

## 0. 离线实验（`harness_selfcheck`）

离线脚本策略运行，可重放：

```bash
PYTHONPATH=src python3 -m evomemory.runner --modes ABCD --llm scripted --per-family 6 --out artifacts
```

原始记录：`artifacts/run_<timestamp>/results.json`、每个模式的 `memory.db`、`summary.md`。

> **这些数字不能用来声称"记忆提升了大模型的任务成功率"。** 策略是确定性规则跟随器，不是模型。它证明的是管线闭合、门控生效、指标可计量。真实模型上的结果见 §7。

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

---

## 7. 真实模型实验（`live_model`）

运行时间 2026-11-21。模型 `deepseek-flash`（temperature=0）。任务数 6（2 family × 3）。
原始记录：`artifacts/live_abcd/`（`mode_A/`、`mode_D/` 各自的 `memory.db` + 证据目录 + `results.json`）。

### 7.1 为什么只跑了 A 和 D

调用预算是 200 次。A（60 次）与 D（104 次，含 4 次失败重试）合计已用 164 次，加上前期探测共 183 次。
**B 和 C 没有跑**：预算不足以再跑完一个模式。C 在本任务集上没有图像、结构上等于 B（见 §2），本来也不提供信息。

这是一个预算约束导致的实验设计妥协，不是"B/C 结果不好所以不报"。

### 7.2 结果

| 模式 | 成功率 | retrieval hit | 负迁移率 | token/任务 | 工具调用/任务 | 准入经验 |
|---|---|---|---|---|---|---|
| A 无记忆 | **0/6 = 0.0%** | — | 0.0% | 9,140 | 10.00 | 0 |
| D 完整生命周期 | **2/6 = 33.3%** | 4/6 | **16.7%** | 15,929 | 12.00 | 6 |

逐任务：

```
escalation_00  fail  inj=0     ← 无记忆，必然失败
escalation_01  OK    inj=1     ← 注入 1 条 escalation 经验
escalation_02  OK    inj=2
closure_00     fail  inj=0     ← 跨 family 经验被 hard gate 拒绝，无记忆可用
closure_01     fail  inj=2     ← 注入了 1 条 closure + 1 条 escalation（见 7.5 的 bug）
closure_02     fail  inj=1     ← 注入 1 条 closure，仍然失败
```

### 7.3 成本

记忆让每条任务的 token 从 9,140 涨到 15,929（+74%），工具调用从 10 次涨到 12 次。
原因是 D 模式每个任务多了 extract（1 次）、critique / reconstruct（每条检索到的经验各 1 次），
而 reasoning 模型的输出 token 里包含看不见的推理 token。

**这与离线结论方向相反**（离线时记忆更便宜），说明"记忆省不省钱"高度依赖任务形态：
离线脚本策略靠记忆减少了重试步骤，真实模型则要为读写记忆额外付推理成本。

### 7.4 这些数字不能证明什么

- 6 个任务的 MDE 是 **±80.9 个百分点**。A 的 0% 与 D 的 33.3% 的差异**远低于**可检出阈值。
- 结论只能是：**"本次运行没有检出记忆的效应"**，不能写成"记忆有效"或"记忆无效"。
- 要做有意义的效应量估计，至少需要 50-100 个 cell（MDE → ±20pp 以内）。

### 7.5 真实运行暴露出的三个问题（这是本次实验最有价值的部分）

**问题 1：reasoning 模型会把输出预算全部花在推理上。**
`deepseek-flash` 的响应同时包含 `reasoning_content` 与 `content`。早期 `max_tokens=300` 时，
对话变长后模型把全部预算用于推理，`content` 返回空串，`finish_reason='length'`。
原实现把它当成"模型拒绝回答"，静默浪费一步。**已修**：`DeepSeekClient` 现在对
"空 content + 非零 completion tokens" 抛带诊断信息的 `LLMError`；动作调用预算提到 1200；
并且**截断历史**（最多 9 条消息），因为推理开销随上下文增长。

**问题 2：模型写出的经验比启发式提取器更抽象，而抽象会丢掉操作数。**
启发式提取器会写"do not act on records with tier=gold and note=hold"——具体、可执行、能迁移。
模型写的是"Before bulk-mutating records ... inspect each candidate's full attribute set for exception markers (special tiers, audit/flag notes)"——**更漂亮，但没有说出 gold 和 audit**。
结果是 escalation 迁移成功（2/3），closure 三次全败：模型知道了"要找例外标记"，但不知道例外标记**是哪些值**。

这是一个真实的**抽象—迁移权衡**，也是我原本没有预料到的：提炼得越"通用"，可执行性越差。

**问题 3：结构化的校验门无法识别语义错误的经验。**
模型写下并被准入了一条经验：*"When a task defines scope with an explicit filter, apply exactly
that filter to every match — **do not invent unstated exclusion rules** from incidental record
attributes."*

这条经验是**错的**——它把"遵守隐藏的排除规则"误学成"不要臆造排除规则"。而校验器的 9 项
检查（证据可解析、非重复、无泄漏、适用条件已声明……）**全都是结构性检查，没有一项能发现它语义错误**。
它至今仍是 `active` 状态。

这直接印证了项目最初的问题 5（如何防止错误记忆）：**程序化校验能挡住"不可复核"的经验，
挡不住"可复核但结论错"的经验**。后者需要下游任务上的结果验证（held-out 上的配对比较），
而那正是 `NEXT_STEPS.md` §4 里尚未接线的部分。

### 7.6 一个真实运行才暴露的 bug（已修复）

§7.2 里 D 模式的负迁移率 16.7% 本不该出现——`MemoryCritic` 在 family 不匹配时会 `REJECT`。
但 `closure_01` 确实注入了一条 escalation 经验：

```
retrieved: [ ... , {"experience_id": "exp_890a9112…", "verdict": "apply"} ]
经验 family = escalation，任务 family = closure
```

原因是 `_llm_critique` **用模型判定替换了启发式判定**，而不是受其约束。类文档写着"模型只能让判断更保守"，
代码却没有实现这个约束——正是我在审计 Recuris 时批评的那类"声明了但没接线"的问题，而我自己的代码里也有一处。
模型很乐意把 escalation 经验判给 closure 任务，于是那条错误经验（问题 3）跨了 family，直接导致 `closure_01` 失败。

**已修**：`Critique` 增加 `hard` 字段；family 不匹配是 hard gate，**不咨询模型**直接拒绝。
新增回归测试 `test_family_mismatch_cannot_be_overridden_by_the_model`（断言模型调用次数为 0）
与 `test_model_may_still_veto_within_a_family`。

**必须说明**：§7.2 的数字是在这个修复**之前**跑的。修复后 D 的负迁移率应当为 0，但预算已用 183/200，
没有重跑。修复本身有测试覆盖，但**它的效果没有在这次实验中被验证**。

### 7.7 结论

真实模型实验**成功验证了管线在 API 模式下可运行**（含预算硬截断、断点续跑、provider 计量入账），
并**暴露了三个在离线脚本策略下不可能发现的缺陷**。但它**没有**产出任何关于"记忆是否提升任务成功率"的可信结论——
样本量差一个数量级。

---

## 8. 扩大样本后的真实模型实验（18 任务，`live_model`）

2026-11-21。模型 `deepseek-flash`。任务数 18 = **3 个 family × 6**（新增 `conditional` family，规则形状是**合取**而不是单属性排除）。
模式 A / B / D。原始记录：`artifacts/live3/`。

预算已放开，本次共消耗约 **1,509 次 API 调用 / 1.49M token**（含一次作废的重跑，见 §8.3）。

### 8.1 结果

| 模式 | 成功率 | retrieval hit | 负迁移率 | token/任务 | 工具调用/任务 | 准入经验 |
|---|---|---|---|---|---|---|
| A 无记忆 | **0/18 = 0.0%** | — | 0.0% | 10,701 | 11.22 | 0 |
| B 简单记忆 | **11/18 = 61.1%** | 88.2% | **0.0%** | 10,476 | 8.61 | 18 |
| D 完整生命周期 | **10/18 = 55.6%** | 83.3% | **0.0%** | 12,415 | 8.67 | 17 |

配对检验（同 18 个任务，McNemar 精确检验）：

| 对比 | 增益 | 损失 | delta | 95% CI | p |
|---|---|---|---|---|---|
| B vs A | 11 | 0 | +61.1% | [+38.9%, +83.3%] | **0.0010** |
| D vs A | 10 | 0 | +55.6% | [+33.3%, +77.8%] | **0.0020** |
| D vs B | 0 | 1 | −5.6% | [−16.7%, 0.0%] | 1.0000 |

按 family 拆开：

| family | 规则形状 | A | B | D |
|---|---|---|---|---|
| escalation | 两个独立的单属性排除 | 0/6 | **5/6** | **5/6** |
| closure | 两个独立的单属性排除 | 0/6 | **5/6** | **5/6** |
| conditional | **合取**：tier=gold **且** note=audit | 0/6 | 1/6 | **0/6** |

### 8.2 三条结论

**结论一：记忆对真实模型有大幅且统计显著的提升——但只在单属性规则上。**
两个单属性 family 从 0/6 提升到 5/6，配对检验 p 在 0.001 量级。这不是脚本策略的数字，是真实模型、真实计费、18 个 cell 的配对结果。

**结论二：完整生命周期（D）没有跑赢朴素记忆（B）。**
11 vs 10，配对检验 p=1.0，置信区间 [−16.7%, 0]。D 每个任务还多花 1,939 token（+18.5%），因为它要做 extract、critique、reconstruct 三次额外模型调用。
唯一的分歧任务是 `conditional_05`，B 成功、D 失败。

也就是说：**在这一组任务上，"校验 + critique + 重建"这套机制的价值没有被证明**。离线脚本实验里 D 明显更好的那些结论，在真实模型上不成立。

**结论三：合取规则会把两个模式一起打垮，而且记忆会让它更糟。**
`conditional` family 里 A 是 0/6（没有记忆，必然失败），B 是 1/6，D 是 0/6。
D 的 5 次失败全部是 **"targets_actioned 未完成"**（少做了 r04/r05），即**过度排除**：模型把 lesson 里的
"exclude records with tier=gold **and** note=audit" 读成了"排除 gold 的**或** audit 的"，于是把只满足一半条件的
诱饵记录也跳过了。

这是本项目最有价值的一条实证：**用自然语言表达规则会丢掉逻辑连接词**。
`tier=gold and note=audit` 与 `tier=gold or note=audit` 在文本上只差一个词，但对任务的影响是决定性的。
上一轮实验（§7.5 问题 2）发现的"抽象丢掉操作数"和这里是同一个病的两个症状。

### 8.3 这一轮里修掉的两个 bug

**bug 1：检索没有按 family 过滤。**
第一个 18 任务版本里，`conditional` family 在 D 模式下**每一个任务的注入都是 0**——
但该 family 明明有 6 条 active 经验。原因：`Harness.retrieve` 调用 `index.search(goal, top_k)` 时没传 family，
而 conditional 的 goal 文本与 escalation 几乎一样（只差隐藏规则），于是 top-3 全被 escalation 的经验占满，
critic 再把它们逐条以 family mismatch 拒绝，任务最终在**零记忆**状态下运行。

同一个 bug 在 B 模式下表现为**相反的错误**：没有 critic，跨 family 的经验被原样注入，造成负迁移。

修复：`index.search(..., family=task.family)`。修复前的那一轮数据完整保留在 `artifacts/live3_prefamfix/`，
**约 658 次调用 / 658k token 花在了一个有 bug 的实验上**——这是真实的成本，记录在此。

**bug 2：模型写出的经验丢操作数。**
这一处引用了未定义函数 `_cites_operand` 导致崩溃。补齐实现后：若模型给出的 lesson 没有点名任何一个
判别性取值（如 `gold`、`audit`），就丢弃它并回落到确定性提取器（后者一定会点名）。
`prompt` 也从 v1 升到 v2，明确要求"NAME THE DISCRIMINATIVE ATTRIBUTE VALUES"。

### 8.4 对 §7 结论的修正

§7 报告过"D 的负迁移率 16.7%，修复 critic 后应为 0"。现在有了更大样本：**修复后 B 和 D 的负迁移率都是 0%**。
原因不是 critic 变强了，而是 **family 过滤在检索层就把跨族经验挡掉了**——critic 根本不需要触发。

所以 §7 里"critic 抑制负迁移"这个说法应当降级为：**在这次任务集上，负迁移是被检索层的 family 过滤消除的，
critic 的独立贡献未被观测到**。

### 8.5 成本

| 项 | 值 |
|---|---|
| A 模式 | 216 次调用 / 198k token |
| B 模式 | 192 次调用 / 212k token |
| D 模式 | 260 次调用 / 245k token（含 12 次空响应重试） |
| 作废的首轮（retrieval bug） | 658 次调用 / 658k token |
| 更早的 6 任务实验（§7） | 164 次调用 / 157k token |
| **合计** | 约 **1,509 次调用 / 1.49M token** |

### 8.6 这些结果仍然不能说明什么

- n=18，MDE ±46.7pp。虽然配对检验给出了 p=0.001 的强证据，但**效应量的置信区间很宽**（+38.9% 到 +83.3%）。
- 只有 3 个 family，其中 1 个（conditional）两个记忆模式都基本失败，所以"B ≡ D"的结论主要由 2 个 family 支撑。
- `C` 模式仍未运行：本任务集没有图像，它与 B 在结构上相同，跑它只会重复 B 的数字。这一点是可验证的（B 与 C 在离线环境下逐位相同），但没有在 live 模式下花 200 次调用去实证。
- 所有任务都是同一个合成环境下的记录表操作，**不能外推到真实业务任务**。
