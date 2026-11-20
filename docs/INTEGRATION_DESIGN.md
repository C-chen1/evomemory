# EvoMemory 统一架构设计

目标不是再做一个 memory 框架，而是把「经验从哪来、凭什么可信、什么时候能被用、用错了怎么收回」这四件事做成**可执行、可审计、可回滚**的闭环，并且能在没有 GPU、没有在线模型的情况下验证这条闭环。

设计约束来自三处：`docs/REPO_AUDIT.md` 的审计结论、`docs/ARCHITECTURE_COMPARISON.md` 的吸收矩阵、以及本机实际能力（CPU only、无 Windows 访问、API key 需外部注入）。

---

## 1. 六个模块与它们的缝

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. Execution Harness          harness.py                            │
│    AgentLoop / TraceRecorder / MemoryGate / ContextManager          │
│    ReplayRunner / Recovery                                          │
├─────────────────────────────────────────────────────────────────────┤
│ 2. Multimodal Evidence Store  evidence.py + multimodal.py           │
│    content-addressed files, IMAGE/THUMBNAIL/TEXT/ENV_STATE          │
├─────────────────────────────────────────────────────────────────────┤
│ 3. Experience Memory          memory.py + db.py                     │
│    versioned rows, state machine, append-only event log             │
├─────────────────────────────────────────────────────────────────────┤
│ 4. Retrieval & Reconstruction retrieval.py + lifecycle.py           │
│    BM25 + hashed dense + RRF; Critic; Reconstructor                  │
├─────────────────────────────────────────────────────────────────────┤
│ 5. Evolution & Validation     lifecycle.py (Validator) + metrics.py │
│    programmatic checks, paired bootstrap gate, MDE                  │
├─────────────────────────────────────────────────────────────────────┤
│ 6. Benchmark & Evaluation     bench.py + runner.py + metrics.py     │
│    deterministic ops env, programmatic verification, A/B/C/D        │
└─────────────────────────────────────────────────────────────────────┘
```

模块之间只有四个方向的依赖：harness 调 lifecyle 与 retrieval；lifecycle 调 evidence 与 store；bench 与 harness 解耦（bench 只提供任务和验收函数）；llm 是最底层的可替换依赖。

### 1.1 关键接口

| 接口 | 位置 | 输入 → 输出 |
|---|---|---|
| `Harness.run_task(bt, run_id)` | `harness.py` | `BenchTask` → `TaskRunResult` |
| `MemoryGate.propose/admit` | `harness.py` | `Experience` → 已校验并激活 / 拒绝 |
| `ContextManager.build(retrieved)` | `harness.py` | 候选经验 → `MemoryBlock`（受 token 预算约束） |
| `ExperienceStore.revise(id, lesson=...)` | `memory.py` | 旧版本 → 新版本 + 旧版本 superseded |
| `EvidenceStore.put(bytes, kind)` | `evidence.py` | 字节 → `Evidence`（内容寻址，可去重） |
| `ExperienceIndex.search(query, k, family)` | `retrieval.py` | 文本 → `[(Experience, score, {channel: score})]` |
| `MemoryCritic.critique(exp, task)` | `lifecycle.py` | 经验 + 当前任务 → `Critique(apply/adapt/reject)` |
| `ExperienceReconstructor.reconstruct(...)` | `lifecycle.py` | 经验 + 任务 → 文本 或 `None` |
| `ExperienceValidator.validate(exp, ...)` | `lifecycle.py` | 候选 → `ValidationReport(passed, checks)` |
| `verify(bt, env)` | `bench.py` | 环境状态 → `Outcome(program-verified)` |

---

## 2. 核心生命周期

```
Task
 ↓  bench.task_env()                      建出一个确定的工具环境
Agent Execution                            harness.AgentLoop：模型只发 action JSON
 ↓  TraceRecorder.record()                每步落 trace_steps（含 receipt）
Outcome Verification                       bench.verify()：逐项比较环境状态，verified_by=program
 ↓  只有 program-verified 才继续           （模型自评的成功一律丢弃）
Experience Extraction                      lifecycle.ExperienceExtractor
 ↓  模型路径失败则回落到确定性提取器        lesson / failure_signature / applicability
Memory Candidate                           status = candidate
 ↓  MemoryGate.propose()                   只有 EXTRACTOR 能建候选
Experience Validation                      ExperienceValidator：9 项程序化检查
 ↓  accept / reject                        模型只能 veto，不能准入
Versioned Memory Store                     status = accepted → active
 ↓
New Task
 ↓  ExperienceIndex.search()               记录 memory_events（含每通道分数）
Memory Retrieval                           top-k，可带 family 过滤
 ↓  MemoryCritic.critique()                apply / adapt / reject
Experience Reconstruction                  reconstruct() → 文本 或 None
 ↓  ContextManager.build()                 按 token 预算裁剪，记录被丢弃的原因
Agent Action                               同一套 AgentLoop
 ↓
Outcome Verification                       ← 回到顶部，闭环
```

### 2.1 每一步「谁有权写」

| 状态 / 数据 | 唯一有权写入者 | 越权后果 |
|---|---|---|
| `experiences.status = pending_validation` | `EXTRACTOR` / `EVOLUTION` | `MemoryWriteError` |
| `= accepted` / `= rejected` | `VALIDATOR` | `MemoryWriteError` |
| `= active` | `GATE` | `MemoryWriteError` |
| `= superseded` / `= deprecated` | `EVOLUTION` / `GATE` / `HUMAN` | `MemoryWriteError` |
| 任务是否成功 | `bench.verify`（程序） | 无法由模型写入 |
| evidence 文件 | 无（写入即不可变） | 只能新增派生证据 |
| `llm_calls` 台账 | `BudgetedLLM` | 每次调用无条件写 |

这套权限是 `memory.py` 里的 `TRANSITION_ACTORS` 表 + `assert_transition()`，测试 `test_model_cannot_accept_or_activate` 直接盯着它。

---

## 3. 数据模型

### 3.1 Experience（每个版本一行，内容不可变）

| 字段 | 作用 | 回答项目简报里的哪个问题 |
|---|---|---|
| `experience_id` | 稳定身份，跨版本不变 | 这条经验是什么 |
| `version` | 递增版本号 | 被改过几次 |
| `task_id` / `trace_id` | 来源执行 | 来自哪次执行 |
| `evidence_refs` | 指向不可变证据 | 凭什么这么说 |
| `action_trace_refs` | `step:<i>:<tool>` | 当时做了什么 |
| `outcome` | success / failure | 结果如何 |
| `failure_signature` | 可复现的失败指纹 | 同类失败如何识别 |
| `lesson` | 可复用结论 | 学到了什么 |
| `applicability_conditions` | `task_family:…` / `tag:…` | 什么时候适用 |
| `verification_status` | 只有 validator 能写 | 是否被验证 |
| `status` | 状态机当前值 | 现在能不能用 |
| `supersedes` | 被本版本取代的版本号 | 怎么防止错误经验继续生效 |
| `source_kind` / `prompt_version` | heuristic / llm；提示词版本 | 可追溯性 |
| `confidence` | 提取时的自评（不参与准入） | —— |

### 3.2 状态机

```
candidate ──submit──> pending_validation ──validator pass──> accepted ──gate──> active
    │                        │                                                   │
    └──────reject────────────┴──────reject──────> rejected                         │
                                                                                   │
                                              superseded <──── revise() ──────────┤
                                              deprecated <──── 人工/进化 ─────────┘
```

`rejected` / `superseded` / `deprecated` 均为终态：不可再被召回，但永远可读。

### 3.3 证据

- 路径 = `evidence/<sha256[:2]>/<sha256[2:]>`，写一次再不改。
- 缩略图、caption 各自是独立 evidence 行，`derived_from` 指向原图。
- 因此「这条经验引用了哪张图、那张图的缩略图是什么、caption 是谁生成的」三层都可追。

### 3.4 审计表

`experience_events`（状态变化的 every change）、`memory_events`（每步检索了什么、注入什么、拒绝了什么及原因）、`llm_calls`（每次调用）、`outcomes`（程序验收结果）。四张表都是只追加。

---

## 4. Harness 的职责边界

明确不做的事：

- **不替模型决策**。AgentLoop 只解析 action JSON、执行工具、把 observation 贴回去。
- **不信任模型的自评**。`verify()` 只读环境状态；模型说"完成了"没有任何权重。
- **不允许模型写状态**。见 2.1。
- **不做隐式压缩**。ContextManager 的预算裁剪每次都记录被丢弃的 id 与原因（`MemoryBlock.rejected`）。

明确要做的事：

- **Replay**：给定 trace，重新执行记录中的工具调用，比对最终状态。报告里写明限制——它验证的是 harness 的确定性，不是模型决策的可复现性。
- **Recovery**：进度从数据库派生（有 trace 且 status=completed 且写了 outcome 才算完成），没有第二份 checkpoint 文件可以与之不一致。

---

## 5. 与三个参考仓库的对应关系

| 借鉴的机制 | 来源 | 本实现里的位置 | 与来源的差别 |
|---|---|---|---|
| 写权限三通道（model/harness/oracle） | Recuris `wm/ledger.py` | `Actor` + `TRANSITION_ACTORS` | 扩展为 6 个 actor，覆盖 validator/gate |
| 准入是算术 | Recuris `gates.py` | `metrics.mde_binary` + 校验器 | 本项目的准入是**程序化检查**为主，统计 gate 用于批次级比较 |
| 记忆块可定位/可替换/可删除 | MemHarness `<memory>` splice | `MemoryBlock` + `rejected` 记录 | 记录拒绝原因，不静默删除 |
| `<EMPTY>` 显式拒绝 | MemHarness | `Reconstructor` 返回 `None` | 拒绝率作为一等指标统计 |
| 检索表示 ≠ 执行证据 | CMA（思想，非代码） | `THUMBNAIL` 与 `IMAGE` 分表 + `derived_from` | 原图永不覆盖 |
| 按会话切分防泄漏 | CMA（思想） | 任务 family 切分 + `leak_guard` | 由 validator 的 `no_eval_leakage` 强制 |

**没有复制任何一行第三方代码**；许可证结论见 `docs/REPO_AUDIT.md` 第 3 节。

---

## 6. 持久化与并发

- 单文件 SQLite，WAL，`synchronous=NORMAL`；schema 版本表 + 追加式迁移列表（`db.MIGRATIONS`）。
- 一次 run 一个数据库文件（`artifacts/run_*/mode_X/memory.db`），因此 A/B/C/D 之间不可能串记忆。
- 没有共享后台进程，没有跨进程锁。当前是单进程顺序执行；如果将来并行化，`experiences` 的主键 `(experience_id, version)` 会挡住重复版本，但 `active()` 的"每 id 只有一个活动版本"目前靠代码保证，需要补一个部分唯一索引。

---

## 7. 明确的设计缺口（写在这里而不是藏起来）

1. **没有 embedding 模型**。`HashEmbedder` 是词法特征，不是语义向量；换成真实模型需要实现 `Embedder` 协议（`dim` + `embed`）。
2. **没有视觉语言模型**。`PixelFactsPerceiver` 只给可测事实，`is_semantic=False`；真实 VLM 需要实现 `Perceiver` 协议。
3. **没有真正的跨会话长期记忆**。当前记忆随 run 的数据库生命周期存在；跨 run 复用需要把 `data/evomemory.db` 作为持久库，这条路已经预留（`Settings.db_path`）但未在实验中启用。
4. **没有并行**，没有分布式。
5. **统计 gate 只用于批次比较**，没有接进单条经验的准入路径——单条经验的准入是程序化检查，这是有意的：n=1 时没有统计显著性可言。
