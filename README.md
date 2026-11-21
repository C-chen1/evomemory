# EvoMemory

面向长程任务的**经验记忆 + Agent Harness**。

一个 Agent 做完任务就忘掉一切：同一个规则今天踩坑、明天还踩。EvoMemory 让执行轨迹变成
**可追溯、可验证、可修正**的经验：任务执行 → 程序评分与证据记录 → 候选经验 → 反例回放验证
→ 激活并被后续任务复用 → 有证据时修正或停用（旧版本永久保留）。

<img width="1" height="1" alt="">

## 一分钟跑起来

```bash
# 环境：零运行时第三方依赖；Python 3.10+
pip install -e ".[dev,images]"      # 或 uv sync

# 全部测试（232 项，约 75 秒）
PYTHONPATH=src python -m pytest -q

# 演示 1：经验记忆主线（离线，不调用任何模型，不花钱）
PYTHONPATH=src python scripts/demo.py

# 演示 2：依赖历史的视觉任务（离线）
PYTHONPATH=src python scripts/demo_history.py
```

两个演示都是**脚本策略**驱动，输出里明确标注 `harness_selfcheck / scripted`。
它们验证的是 harness 与任务设计，**不是模型效果**。

### 演示 1 会看到什么

8 个任务、同一 family。前 3 个任务遵循策略 A，从第 4 个起策略**悄悄换成 B**，family 名不变，
所以按 A 学到的经验仍会被检索到。两次运行唯一的差别是"允不允许退休失效的经验"：

```
WITHOUT evolution   2/8   显式跳过 20 次，引用式 strike 0 次，退休 0 条
WITH evolution      5/8   显式跳过 17 次，引用式 strike 2 次，退休 1 条
```

第二组在第 5 个任务恢复，并且能指出**为什么**退休了那条经验：

```
experience exp_8511…  task=escalation_03_118  record=r02
reason: agent explicitly skipped citing this experience; the experience
        exempted the record; grader required it
```

### 演示 2 会看到什么

规则：某个区域**在先前面板里红过 2 次以上**就锁定，必须保持关闭。这个次数不在当前图片里、
不在任务文字里、任何工具也查不到——只能来自历史。

```
image-only   2/6      // 只看当前面板，遇到锁定区必然打开它
with history 6/6      // 用历史计数，全部正确
```

同一个演示还会打印"**像素完全相同、历史不同 → 正确动作不同**"，这是负迁移的最小例子。

## 真实模型实验

```bash
cp .env.example .env      # 填 DEEPSEEK_API_KEY / DEEPSEEK_MODEL

# 付费调用必须走共享账本，否则会被拒绝启动（这是硬约束，不是提醒）
PYTHONPATH=src python -m evomemory.runner --modes AD --llm api \
  --per-family 6 --shared-stage3-budget
```

- 代码不会猜模型 id：`model` 为空直接报错。
- 账本：`artifacts/stage3/round_budget.db`，上限 450 次 / 100 万 token；每个请求先预留后结算。
  用量未知时保留预留并**阻塞后续请求**，不会被记成 0。
- 想绕过账本必须显式加 `--allow-unbudgeted`，并会在 stderr 打出警告。

## 架构

| 模块 | 职责 |
|---|---|
| `harness.py` | Agent 循环、Memory Gate、token 预算上下文、Trace、Replay、Recovery、`check_rule` 可执行规则工具、归因式演化 |
| `memory.py` + `db.py` | 版本化经验库：不可变版本行、7 状态机、6 类写入者权限、追加式事件日志 |
| `predicate.py` | 三值谓词引擎（all_of / any_of / not / 比较符），深度与字段白名单，禁用 `eval` |
| `counterexample.py` | 候选经验在训练证据上的回放验证、结构化差异 |
| `lifecycle.py` | 经验提取（模型 + 确定性双路）、9 项校验、critique、重建 |
| `evidence.py` | 内容寻址证据存储（sha256），派生表示带 `derived_from` 血缘 |
| `bench.py` | 文本任务环境：程序化判分、规则漂移、三种规则形状 |
| `visual_bench.py` / `history_bench.py` | 视觉面板环境；后者依赖历史信息 |
| `shared_budget.py` | 跨进程共享的请求/ token 账本 |

**三条硬约束**（都由代码强制，不是文档约定）：

1. **模型不能宣布成功**——成功只来自 `bench.verify()` 的程序比对。
2. **模型不能写状态**——接受/激活/退休只能由 validator 与 gate 发起。
3. **证据不可覆盖**——按内容哈希寻址，原图永不改写。

## 结果在哪里

```
artifacts/
├── demo/                      演示输出（with_evolution / without_evolution / history）
├── stage3/round_budget.db     共享预算账本（所有付费调用的唯一来源）
├── stage3/panel_v4_*/         视觉预注册实验（未完成，见 docs/STAGE4_RESULTS.md）
└── live3/ live_abcd/ drift*/  历史真实模型实验（保留原始数据，未改写）
```

`<run>/mode_X/memory.db` 里可以直接查：`experiences`（版本）、`experience_events`（状态变更）、
`rule_decisions` / `task_actions` / `skip_decisions`（决策链）、`experience_strikes` /
`attribution_weak` / `attribution_unknown`（归因分级）、`llm_calls`（用量台账）。

## 状态：已实现 / 已测试 / 有观察 / 未证明

| 项 | 状态 |
|---|---|
| 经验生命周期（候选→验证→激活→修订/停用，旧版保留） | 已实现，232 项测试 |
| 反例回放准入、版本事务化切换、恢复一致性 | 已实现，有回归测试 |
| 三值谓词引擎 + 从反例推断连接词 | 已实现，39 项测试 |
| 历史依赖视觉任务（离线） | 已实现，12 项测试证明区分力 |
| 记忆对真实模型有提升 | **有观察**：0/18 → 61%（McNemar p=0.001）；未在更多任务上重复 |
| 显式跳过 + 经验引用在真实模型上可用 | **有观察**：23/23 次跳过都引用了经验（单次运行） |
| 历史任务对真实模型有区分力 | **有观察**：1 个跑完的锁定面板出现预期过度操作（n=1） |
| 完整生命周期优于简单记忆 | **未获支持**：11/18 vs 10/18（p=1.0），成本更高 |
| 结构化表示优于散文 | **未获支持**：6/8 vs 6/8（p=1.0），贵 43% |
| 视觉记忆带来增益 | **未证明**：任务设计本身无提升空间，见下 |

## 已知边界

- **面板任务（panel-v2/v3/v4）没有研究价值**：文字要么与判分规则冲突，要么把答案提示出来。
  已停止投入，改用 `history_bench.py`。
- **跨模态检索未实现**：当前是"文本索引 + 视觉证据复用"，不是向量级跨模态 RAG。
- **真实模型实验规模小**：18 个任务、2-3 个规则模板，有效样本量接近模板数。
- **空响应率约 25%**：reasoning 模型会把输出预算吃满。已做诊断与缓解，未根治。
- **G 臂（结构化 + 门，无 critique）未跑**。
- 预算是硬约束：本轮 404/450 次调用后停止付费实验。

## 文档

| 文档 | 内容 |
|---|---|
| `docs/STAGE4_RESULTS.md` | **最新状态**：归因评级、版本事务、历史任务、试点结果 |
| `docs/STAGE3_RESULTS.md` | 上一阶段：4 个缺陷修复、回放准入、面板任务的天花板发现 |
| `docs/DECISIONS.md` | 设计决策与代价 |
| `docs/FAILURE_ANALYSIS.md` | F1–F6 根因与处置 |
| `docs/ABLATION_RESULTS.md` | 消融实验（含被否决的机制） |
| `docs/BLOCKERS.md` | 未解决事项与解除条件 |
| `docs/RESUME_TECHNICAL_REPORT.md` | 面试视角的技术报告 |
