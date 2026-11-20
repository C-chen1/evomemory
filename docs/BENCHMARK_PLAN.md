# Benchmark 计划

本文写的是**实验设计**，包括什么算证据、什么不算。结果在 `docs/EXPERIMENT_RESULTS.md`。

---

## 1. 要回答的问题

本项目不试图回答"记忆系统是否普遍有用"（这需要大模型 + 大基准）。它回答四个更窄、但可以被证伪的问题：

1. **Q1 闭环是否真的闭合**：一次失败的执行能否变成一条可被检索、可被重用的经验，并在后续同类任务上改变行为？
2. **Q2 校验门是否在起作用**：加不加程序化校验（B/C 与 D 的差别），写进库里的经验有什么不同？
3. **Q3 Critique 是否抑制负迁移**：没有 critique 时，跨 family 的错误经验注入会不会真的造成失败（B/C 的 25%）？加上之后是否消失（D 的 0%）？
4. **Q4 成本是否可计量**：每个模式的实际 token 与工具调用是多少？记忆是省钱还是费钱？

不回答：视觉记忆是否提升成功率（需要 VLM，见 `NEXT_STEPS.md`）。

---

## 2. 四种模式

四个模式**共用同一个 runner、同一套任务、同一份预算**，只通过 `RunConfig` 切换：

| 模式 | 检索 | 注入 | Critique | Reconstruction | 写入门槛 |
|---|---|---|---|---|---|
| **A** 无记忆 | ✗ | ✗ | ✗ | ✗ | 不写 |
| **B** 简单向量记忆 | ✓ | 原文 lesson | ✗ | ✗ | 无程序化校验（候选直接 active） |
| **C** 多模态记忆 | ✓（含 caption 通道） | 原文 lesson | ✗ | ✗ | 同 B |
| **D** 完整生命周期 | ✓ | 重建后的 guidance | ✓ | ✓ | 9 项程序化校验 |

设计意图：**B 与 D 的差 = 校验 + critique + reconstruction 的合力**；**C 与 B 的差 = 多模态证据通道**。

⚠️ 在离线条件下 **C ≡ B**，因为测试环境没有图片、caption 通道为空。这不是失败，是预期结果，`EXPERIMENT_RESULTS.md` 里如实写明。任何声称 C 优于 B 的离线结果都应当被怀疑。

---

## 3. 任务集

`bench.build_dataset(seed, per_family)`，两个 family：

| family | 动作 | 隐藏规则（agent 不可见） |
|---|---|---|
| `escalation` | 把 region 内 priority ≥ P 的 open 记录升级 | gold tier 不动；note 含 `hold` 不动 |
| `closure` | 把 region 内 priority ≤ P 的 escalated 记录关闭 | gold tier 不动；note 含 `audit` 不动 |

每个任务的记录表由 `build_table()` **按固定配方构造**：2-3 个必须动作的目标、2 个看起来像目标但被规则排除的记录、3-4 个近似干扰项（错 region / 错阈值 / 已是目标状态）。

早期版本用随机生成，结果出现了**目标集为空**的任务——这种任务什么都不做就能通过，等于往数据里掺噪声。现版本 `test_every_task_has_targets_and_exclusions` 断言每个任务都有目标也有排除项。

### 3.1 为什么第一个任务必然失败

隐藏规则不在 goal 里。agent 第一次遇到 `escalation` 一定会动 gold 记录，从而失败；失败的 `hidden_rule_respected` 检查会把**导致排除的那个属性**写进 detail（`r03(tier='gold')`，只写真正起作用的属性）。提取器把这个属性变成 lesson，第二个同类任务就能成功。

因此**成功率的结构性上限是 `(n_tasks - n_families) / n_tasks`**。12 个任务、2 个 family → 上限 83.3%。D 模式达到 83.3% 就是"每个 family 只失败一次，之后全对"。

---

## 4. 验收方式

`bench.verify()` 逐项比较环境状态，返回 `Outcome(verified_by=program)`：

| check | 含义 |
|---|---|
| `targets_actioned` | 所有应动作的记录到达目标状态 |
| `hidden_rule_respected` | 所有应排除的记录状态未变（detail 指出阻塞属性） |
| `no_collateral_changes` | 无关记录未被修改 |
| `log_written` | 至少写了一条日志 |

四项全过才算成功。没有任何一项依赖模型判断。

---

## 5. 指标定义

| 指标 | 精确定义 | 陷阱 |
|---|---|---|
| `success_rate` | 程序验收通过的任务数 / 总任务数 | 上限受 §3.1 限制 |
| `retrieval_hit_rate` | 在"此前该 family 已有 active 经验"的任务中，至少注入一条同 family 经验的比例 | **不是 Recall@K**：不判断召回的是不是对的那一条 |
| `negative_transfer_rate` | 注入跨 family 经验**且** `hidden_rule_respected` 失败的任务比例 | 两个条件都要求；只是注入了无关记忆但没造成伤害，不算 |
| `evidence_grounding_accuracy` | 证据引用可全部解析的经验 / 全部经验 | 未达 1.0 说明有经验不可复核 |
| `mean_tokens_per_task` | （输入 + 输出）token / 任务数 | 离线是估算（4 字符/token），在线是 provider 回报 |
| `mean_tool_calls` | 工具调用数 / 任务数 | 衡量"想得更多还是做得更多" |

---

## 6. 功效（必须写在结果前面）

`metrics.mde_binary(n)` 给出该样本量下可检出的最小效应。12 个 cell 时 **MDE ≈ ±57 个百分点**。

含义很直接：**在 12 个任务上，只有"从几乎全错变成几乎全对"这种量级的差异才可能被检出**。任何 10-20pp 的差异在这个 n 下都是噪声。所以本项目的 A/B/C/D 对比只能用来验证机制是否按预期工作，不能用来宣称效应量。

要得到可信的效应量，需要至少几百个 cell，且必须用 live model——两者都不在今晚的能力范围内。

---

## 7. 效度威胁（threats to validity）

1. **离线策略是规则跟随器**。它证明管线通，不证明模型会变好。已在结果文件里以 `evidence_class=harness_selfcheck` 标注。
2. **family 只有两个**，且规则形状高度相似（一个 tier 属性 + 一个 note 属性）。换一个规则形状（数值阈值、组合条件）可能表现不同。
3. **第一个任务必失败**的结构性上限，让 `success_rate` 不能与外部基准比较。
4. **token 是估算**，不是 provider 计量。模式间的相对比较有意义，绝对值没有。
5. **单机单进程**，没有并发带来的排队与重试，延迟数据不代表线上。

---

## 8. 判定标准（事先写定，避免事后解释）

| 结论 | 需要看到什么 |
|---|---|
| "闭环闭合" | A 模式 0% 且 D 模式在**每个 family 的后续任务**上全部成功 |
| "校验门有效" | D 的 admitted 数显著少于 B（重复/不合格被挡住），且没有降低成功率 |
| "critique 抑制负迁移" | B/C 的 negative_transfer_rate > 0 且 D = 0 |
| "多模态有增益" | **离线不可能得出**；需要 VLM 接入后 C > B 且差异超过 MDE |

若某项达不到，写"未验证"，不写"部分有效"。
