# EvoMemory 最终报告

项目：EvoMemory — 面向长程任务的多模态自演化经验记忆系统
根目录：`D:\python\PyCharm\Project\06_evomemory`
报告日期：2026-11-21

**状态约定（全文严格遵守）**

- **已实现**：有可运行代码。
- **已验证**：有通过的测试或实测数据支撑。
- **已设计**：有架构文档，代码未完成。
- **待研究**：需要进一步实验，当前无结论。

**一句话结论**：核心经验闭环在离线条件下**已实现并已验证**（88 个测试全绿、A/B/C/D 跑通、结果可逐位复现）；三个参考仓库**已审计**并有实测反例；多模态的**存储与压缩已实现并实测**，但**视觉带来的效果增益待研究**；真实模型上的效应量**待研究**，被 API key 阻塞。

---

## 1. 三个 GitHub 项目的源码审计结论

| 仓库 | Commit | 许可证（实测） | 一句话结论 |
|---|---|---|---|
| Recuris | `7d3745a` | Apache-2.0 | 机器很小（~2.5k 行 kernel），脚手架极大（`driver.py` 11k 行）；多处机制"文档有、代码没接线" |
| MemHarness | `74db61e` | Apache-2.0（README badge 误写 MIT） | 代码里没有 critique/reconstruct 概念；发布配方**未训练** adaptor |
| CMA-Harness | `84732bc` | **无许可证 = 保留所有权利** | 部分发布；"记忆压缩"无 embedding、无 token 量化、无授权 |

**三条可复现的硬发现**（grep 即可验证，不需要跑实验）：

1. **Recuris**：`metaagent run` 的每次评估里 `TAU2_GATE_TERM` / `TAU2_STATUS_BOARD` 都是关闭的（`downstream.py:622,679-695` 过滤掉所有 `TAU2_*` 且未补回），尽管包的 manifest 写着 `status_board: true`。另外 `field_writers`、`leakage_check`、`fingerprint_verify`、`blocked` do-not-repeat ledger 均为声明未落地；仓库**没有 tests 目录**，而多处注释声称有 byte-parity 测试。
2. **MemHarness**：`agent_system/` 全文搜索 `critique|reconstruct` **0 命中**（只在 README 出现）；实际是"检索 + 一次额外的 LLM 改写 pass + `<EMPTY>` 拒绝"。发布脚本 `train_alfworld.sh:255-258` 设 `train_memory_adaptor=false`，而 `mem_adaptor_training.py:331-338` 要求为 true 才训练——改写 token 不进 GRPO 损失。其"pure GRPO 对照"被 `require_memory_retrieve=True` 污染。
3. **CMA-Harness**：全仓库无 LICENSE，`research/THIRD_PARTY_NOTICES.md` 末句原话 *"This file does not assign a new license to the authors' research code."* ——**不能复用其任何代码**，只能独立重实现思想。技术上它是"用 VLM 读全量缩略图线性扫描"，无 embedding、无向量索引、无遗忘、无 `parent_id`，且代码里没有任何 token 成本测量。

**许可证矩阵**：Recuris 与 MemHarness 的代码可复用（Apache-2.0，需保留声明）；其 vendored 的 **SkillFlow 无许可证**；**CMA 完全不可复用**。本项目的实际做法是**一行代码都不复制**，全部独立实现。

---

## 2. 实际复现结果

| 项目 | 结果 | 证据 |
|---|---|---|
| CMA-Harness 的 14 个 CPU 单测 | **通过 14/14**（0.044s） | `docs/REPRODUCTION.md` §2.1，命令可重放 |
| Recuris `held_out_paired_gate` | **行为符合文档，且比文档更严** | 探针输出见 §2.2：`mixed net>0` 情形被 CI 拒绝而非 reg_cap |
| MemHarness `memory_text_dedupe` | **行为符合 docstring** | §2.3；并发现长度不匹配时静默返回 0.0 的隐患 |

**未执行**（逐项原因已记录）：Recuris 的任何 benchmark 运行（需 tau2 + Docker + 在线 judge）、MemHarness 的任何训练/评测（需 8 GPU + vLLM）、CMA 的任何 benchmark（需 8 卡 + 未发布的权重与 M2CA-Bench）。

**因此本项目没有任何"复现了论文数字"的声明。**

一个可迁移的工程经验：本环境下单个 bash 调用有效窗口约 175 秒，任何超过 3 分钟的网络操作必须设计成可续传（`--depth 1` / `--filter=blob:none` / 逐 blob `cat-file`）。CMA 的完整 clone 失败 4 次，最终靠 `--filter=blob:none` 取 tree + 按 SHA 抽取 38 个文本文件成功。

---

## 3. 架构融合方案

六个模块，吸收三家各一处最强机制，**明确不吸收**它们的整体架构：

| 模块 | 吸收自 | 吸收什么 | 不吸收什么 |
|---|---|---|---|
| Execution Harness | Recuris | 写权限三通道；每轮重编译上下文 | 11k 行 driver、tau2/Docker 依赖 |
| Multimodal Evidence Store | CMA（仅思想） | 检索表示 ≠ 执行证据 | 全部代码；原地覆盖原图 |
| Experience Memory | Recuris + 自研 | 卡片式经验 + 来源 | 无卡片级版本、淘汰只能删除 |
| Retrieval & Reconstruction | MemHarness | 可逆记忆块；`<EMPTY>` 显式拒绝；`MemoryEvent` 审计 | Milvus FLAT；"检索即写库" |
| Evolution & Validation | Recuris | 对 item 重采样的配对 bootstrap；MDE | 把 gate 当唯一真理 |
| Benchmark & Evaluation | 三家 + 自研 | 同函数生成两 arm、只切一个变量 | 依赖 LLM judge 的判分 |

三家都没有、必须自己做的三件事：**卡片级版本与显式血缘**、**证据内容寻址**、**预算可计量**。

---

## 4. 当前代码结构

```
06_evomemory/
├── src/evomemory/          14 个模块，~3.4k 行
│   ├── models.py           数据模型 + 状态机 + 权限表
│   ├── db.py               SQLite，版本化 schema，9 张表
│   ├── evidence.py         内容寻址证据存储
│   ├── memory.py           版本化经验库
│   ├── llm.py              预算台账 + 离线/在线客户端
│   ├── retrieval.py        BM25 + 哈希向量 + RRF
│   ├── lifecycle.py        提取 / 校验 / critique / 重建
│   ├── harness.py          六项机制 + Replay + Recovery
│   ├── multimodal.py       附件 / 感知 / 压缩实测
│   ├── policy.py           确定性离线策略
│   ├── bench.py            确定性环境 + 程序化验收
│   ├── metrics.py          指标 + MDE
│   ├── runner.py           A/B/C/D 运行器
│   └── config.py           配置（无硬编码密钥）
├── tests/                  9 个文件，88 个测试，全绿
├── scripts/compression_study.py
├── docs/                   9 份阶段文档 + 3 份逐仓库审计
├── artifacts/              实验结果、每个模式的 memory.db
└── references/             三个仓库（.gitignore 排除）
```

依赖：**零运行时第三方依赖**（纯标准库）。可选 `pillow`（图像）、`pytest`（测试）、`uv`（环境）。

---

## 5. 核心记忆生命周期

```
Task → Agent Execution → Trace（每步含 receipt）→ 程序化验收
     → 经验提取（失败可回落到确定性提取器）→ 候选
     → 9 项程序化校验 → accepted / rejected
     → 激活（只有 GATE 能激活）→ 版本化存储
     → 新任务 → 检索（记录每通道分数）→ Critique（apply/adapt/reject）
     → 重建（可返回 None）→ token 预算裁剪 → 注入 → 行动
     → 验收 → 回到顶部
```

**七问逐条回答**（对应项目简报 7.3 节）：

| 问题 | 系统如何回答 | 落在哪 |
|---|---|---|
| 为什么保存这条经验 | `lesson` + 校验器的 9 项 check | `experiences.verification_status` |
| 来自哪次执行 | `task_id` / `trace_id` / `action_trace_refs` | 三个外键列 |
| 是否经过验证 | `verification_status` 只能由 validator 写 | `TRANSITION_ACTORS` 表 |
| 为什么检索到它 | 每通道分数 + verdict | `memory_events.retrieved_json` |
| 是否适用于当前任务 | `Critique(apply/adapt/reject)` + 理由 | `memory_events` |
| 经验有错如何修改 | `revise()` 产生 v(n+1)，旧版 superseded 且永久可读 | `experiences.supersedes` |
| 如何防止影响未来任务 | 三态终态 + family 不匹配直接 reject + 校验门 | 状态机 + `MemoryCritic` |

---

## 6. 多模态证据组织方式

- **原始证据不可变**：`evidence/<sha[:2]>/<sha[2:]>`，写入后只读。缩略图与 caption 是**独立行** + `derived_from` 边，因此"这条经验引用了哪张图、缩略图是什么、caption 谁生成的"三层可追。
- **感知分层且诚实**：`PixelFactsPerceiver` 只产出可测事实（尺寸、主色、色数），`is_semantic=False`；`SidecarCaptionPerceiver` 用于测试语义通路。真实 VLM 实现同一 `Perceiver` 协议即可接入。
- **压缩有实测**：48 个测量单元（4 图 × 4 尺寸 × 3 缩放）。关键结论——`squash`（CMA 的做法）在 300×900 图上内容宽高比误差达 2.0；`letterbox` 保住几何但全局色彩被黑边稀释（小图缩到大画布时平均色距 245.8）；`crop` 保色但丢视野。**这三条是测量，不是主张。**
- **不做未验证的声明**：`C` 模式在离线环境**逐位等于** `B`（无图像 → caption 通道为空），结果文件里如实写明。

---

## 7. Harness 关键约束

| 约束 | 实现 | 测试 |
|---|---|---|
| 模型不能宣布任务完成 | 成功只来自 `bench.verify` | `test_do_nothing_agent_fails` |
| 模型不能写状态 | `TRANSITION_ACTORS` + `assert_transition` | `test_model_cannot_accept_or_activate` |
| 模型自评的成功不得入库 | `verified_by != PROGRAM` 直接不提取 | `test_extractor_refuses_model_judged_outcome` |
| 模型只能 veto 不能准入 | `model_veto` 只能让 check 失败 | `test_model_veto_can_only_block` |
| 预算硬上限 | 调用**前**检查 | `test_budget_stops_before_the_call_that_would_exceed` |
| 证据不可覆盖 | 内容寻址 + 只写一次 | `test_path_is_content_addressed_and_file_never_rewritten` |
| 结果可复现 | 排序按内容而非 uuid | `test_two_runs_are_identical` |
| 可重放 | `ReplayRunner`（只重放工具调用） | `test_replay_reproduces_the_recorded_outcome` |
| 可恢复 | 进度从 DB 派生，无第二份 checkpoint | `test_recovery_resume_runs_only_the_remainder` |

---

## 8. 测试通过情况

**88 个测试，全部通过**（`PYTHONPATH=src python3 -m pytest -q`）：

| 文件 | 数量 | 覆盖 |
|---|---|---|
| `test_memory.py` | 11 | 版本、状态机、写权限、不可变性 |
| `test_evidence.py` | 7 | 内容寻址、去重、派生血缘 |
| `test_llm_retrieval.py` | 12 | 预算、BM25、RRF、余弦长度校验 |
| `test_lifecycle.py` | 13 | 提取、9 项校验、critique、重建 |
| `test_harness.py` | 12 | 预算裁剪、gate、A/D 模式 |
| `test_bench.py` | 11 | 环境、验收、规则反馈 |
| `test_multimodal.py` | 12 | 附件、感知、压缩、保真度 |
| `test_replay_recovery.py` | 7 | 重放、恢复 |
| `test_determinism.py` | 3 | 两次运行逐位相同 |

开发中修掉的一个真 bug：检索排序的 tie-break 原本用 uuid4，导致两次相同运行结果不同（B=50% vs C=58%，而两者本应完全相同）。现按内容文本决胜负（见 `DECISIONS.md` D11）。

---

## 9. Benchmark 实际结果

**证据等级 `harness_selfcheck`（离线脚本策略）**。12 个任务，MDE ≈ ±57pp。

| 模式 | 成功率 | retrieval hit | 负迁移率 | token/任务 | 工具调用 | 准入经验 |
|---|---|---|---|---|---|---|
| A 无记忆 | 0.0% (0/12) | — | 0.0% | 4232 | 7.50 | 0 |
| B 简单向量记忆 | 58.3% (7/12) | 100% | 25.0% | 3813 | 6.08 | 12 |
| C 多模态记忆 | 58.3% (7/12) | 100% | 25.0% | 3813 | 6.08 | 12 |
| D 完整生命周期 | 83.3% (10/12) | 100% | 0.0% | 3399 | 5.83 | 4 |

三条可辩护的机制层结论：

1. **D 达到该任务集的结构性上限**（`(12-2)/12`），即每个 family 只失败一次、之后全对。
2. **Critique 消除负迁移**：B/C 的 25% → D 的 0%。
3. **校验门控制记忆库规模**：同样 12 个候选，D 只准入 4 条（8 条被判重复），且成功率更高、成本更低。

一个反直觉但重要的观察：**B 模式会随时间变差**——`escalation_05` 在连续成功 4 次之后又失败，因为 12 条 active 经验里大多是"这次成功了"的同义重复，把真正有用的规则经验挤出了 top-k。这为 MemHarness 在发布脚本里关闭去重（`train_alfworld.sh:135-136`）提供了一个具体代价。

### 9.1 真实模型实验（`live_model`，18 任务）

模型 `deepseek-flash`（reasoning 模型），3 个 family × 6 = 18 个任务，模式 A / B / D，约 1,509 次调用 / 1.49M token。

| 模式 | 成功率 | 负迁移 | token/任务 |
|---|---|---|---|
| A 无记忆 | 0/18 = 0.0% | 0.0% | 10,701 |
| B 简单记忆 | 11/18 = 61.1% | 0.0% | 10,476 |
| D 完整生命周期 | 10/18 = 55.6% | 0.0% | 12,415 |

配对 McNemar：**B vs A p=0.0010**，**D vs A p=0.0020**，**D vs B p=1.0（无差异）**。

**结论一（正面）**：记忆对真实模型有大幅且显著的提升——0% → 约 58%。这是真实模型、真实计费、配对检验的结果，不是脚本策略。
**结论二（负面）**：完整生命周期 D **没有**跑赢朴素记忆 B（11 vs 10），且每任务多花约 1.9k token。
在这组任务上，"校验 + critique + 重建"这套机制的价值未被证明。离线脚本实验里 D 明显更好的结论在真实模型上不成立。
**结论三（最有价值）**：新增的**合取规则** family 把两个记忆模式一起打垮——A 0/6、B 1/6、D 0/6。
D 的失败全是"少做了记录"，即把 lesson 里的 `tier=gold **and** note=audit` 读成"或"，把只满足一半条件的诱饵也跳过。
**用自然语言表达规则会丢掉逻辑连接词**，这是比"抽象丢操作数"更根本的表示层缺陷。

这一轮还修掉两个 bug：检索没有按 family 过滤（导致 conditional 全程零注入，作废了一轮 658 次调用）；模型经验丢操作数时回落到确定性提取器。
并且**修正了 §9.1 上一版的结论**：负迁移归零不是 critic 的功劳，而是检索层 family 过滤的功劳，critic 的独立贡献未被观测到。

**不得引用为**"记忆提升了大模型成功率"的**泛化**结论——它成立于"单属性排除规则"这一种任务形状，在合取规则上直接失效。

---

## 10. 未完成事项

| 事项 | 状态 | 阻塞 |
|---|---|---|
| live-model A/B/D（18 任务） | **已完成，有统计结论** | B/D 显著优于 A（p=0.001/0.002）；D 未优于 B（p=1.0） |
| live-model C | 未运行 | 本任务集无图像，C 在结构上等于 B；离线已验证 B≡C |
| 合取/逻辑连接词任务 | **已测，失败** | `conditional` family 两模式均 ≈0/6；需要结构化表示（见 §11） |
| 扩大样本到更窄 CI | 待研究 | 当前 CI 为 +38.9%~+83.3%，需要 3-5 倍 cell 数 |
| 视觉带来的成功率增益 | 待研究 | 无 GPU、无 VLM |
| 三个参考仓库的 benchmark 复现 | 未执行 | Docker / 8 卡 / 未发布权重 |
| 跨会话长期记忆 | 已设计 | 未接进实验 |
| 统计 gate 接入批次级演化 | 已设计 | 单条准入目前是程序化检查（有意为之） |
| 大规模任务集（50-100 cell） | 已设计 | 需要更多规则形状的 family |

API 用量：**约 1,509 次调用 / 1.49M token**（provider 回报计量）。其中约 658 次调用 / 658k token 花在一个后来发现检索 bug 的作废实验上，如实记录。

---

## 11. 下一阶段技术研究方向

1. **把经验从自由文本升级为"结构化谓词 + 自然语言"双表示**（最有优先）。18 任务实验直接证明：`tier=gold and note=audit` 被读成"或"，合取 family 两个模式全军覆没。修复方向是让 `Experience` 携带一个可机器求值的谓词（如 `{"all_of": [{"field":"tier","eq":"gold"},{"field":"note","eq":"audit"}]}`），自然语言只作解释，装配上下文时两者一起注入。这也是唯一能让"经验"从"提示片段"变成"可执行知识"的路径。
2. **压缩对生成侧的影响**（相对 CMA 最有区分度）：它只压缩检索、生成仍用原图，从未回答"如果生成也只能看压缩表示会怎样"。现有 `fidelity()` 已能测几何与色彩偏差。
3. **记忆污染的时间曲线**：离线数据里 B 模式在任务 5 退化；live 数据里 B 的 18 条经验里有大量同义重复。值得量化"active 经验数 vs 检索命中正确规则的概率"。
4. **critic / validator 的独立贡献**：18 任务实验里 D 没有跑赢 B，说明这套机制目前是成本而非收益。要证明它的价值，需要一个"记忆本身会变坏"的场景（过期规则、近似错误规则），而不是现在这种"规则一旦学到就不会错"的场景。
5. **统计 gate 与零方差**：Recuris 的 gate 在 item 差异完全一致时给出零宽度 CI，过度自信。自实现时必须显式处理。

---

## 12. 可以真实写入简历的个人贡献

以下每一条都能指向代码或数据，且不夸大：

- **独立完成三个前沿 Agent 记忆系统的源码审计**（Recuris / MemHarness / CMA-Harness），产出 1000+ 行带 file:line 证据的审计报告，发现并记录了两处"论文声称但发布代码未接线"的关键缺口（Recuris 的 harness 机制在演化评估中被关闭；MemHarness 的 adaptor 未被训练），以及一处许可证风险（CMA-Harness 无 LICENSE 且明确声明不授权）。
- **设计并实现了一个版本化的经验记忆系统**：经验按 `(experience_id, version)` 不可变存行，状态机区分 7 种状态、6 类写入者，模型的写权限被结构性限制为"只能提议"；每次状态变化、每次检索、每次 LLM 调用都有独立审计表。
- **实现了内容寻址的多模态证据层**，原图写入后不可变，缩略图与 caption 作为派生证据带 `derived_from` 血缘，从结构上排除了"原地覆盖证据"与"缓存键不含内容"两类真实缺陷。
- **搭建了不依赖 LLM judge 的确定性 benchmark**（程序比对环境状态），并实现 A/B/C/D 四模式的同任务同预算对照，结果逐位可复现。在 12 个任务上得到：无记忆 0%、简单记忆 58%（25% 负迁移）、完整生命周期 83%（0% 负迁移、准入经验从 12 条压到 4 条）。
- **发现并修复了一个破坏实验复现性的缺陷**：检索排序以 uuid 作 tie-break，导致两次相同运行结果不同；改为按内容决胜负后两次运行逐位一致。
- **工程实现**：88 个测试全绿，零运行时第三方依赖，全部基础设施（BM25、RRF、bootstrap、内容寻址存储、SQLite schema 迁移）自研。

- **在真实模型上完成了 18 任务的配对对照实验**（3 种任务规则 × 6 个任务，A/B/D 三模式，约 1,509 次 API 调用 / 1.49M token）：记忆把成功率从 0%（0/18）提升到 61.1%（B）与 55.6%（D），配对 McNemar 检验 p=0.0010 / 0.0020；同时得到明确的**负面结论**——完整的校验+critique+重建链路并未优于朴素记忆（11 vs 10，p=1.0），且每任务多花 1.9k token。
- **用新增的合取规则任务证明了表示层缺陷**：当隐藏规则是 `tier=gold 且 note=audit` 时，两个记忆模式都退化到 0-1/6，模型把"且"读成"或"并过度排除。这条实证把一个抽象担忧（"自由文本经验会丢结构"）变成了可复现的实验结果，并直接决定下一步方向（结构化谓词 + 自然语言双表示）。
- **发现并修复了自己实现里的三个缺陷**：检索未按 family 过滤导致正确经验永远进不了 top-k（作废一轮 658 次调用，如实计入成本）；模型经验丢失判别性操作数时未回落到确定性提取器；`MemoryCritic` 允许模型覆盖跨族硬约束。三者都有回归测试。

**不写入简历的**：任何泛化的"记忆提升了 X% 成功率"表述——该结论只在单属性排除规则上成立，在合取规则上直接失效；任何"复现了论文结果"的表述。
