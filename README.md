# EvoMemory

**文档知识库 RAG + Agent 经验记忆。**

两条并列的主线：**知识库**负责从文档里检索事实并给出可追溯引用；
**经验记忆**负责把任务中的可复用经验沉淀下来、验证并按版本演化。两者数据分开、职责分开。

一个 Agent 做完任务就忘掉一切：同一个规则今天踩坑、明天还踩。EvoMemory 让执行轨迹变成
**可追溯、可验证、可修正**的经验：任务执行 → 程序评分与证据记录 → 候选经验 → 验证与复用
→ 有证据时修正或停用（旧版本保留）。具备结构化条件和样本的候选可进行反例回放；
未声明策略作用域时，缺少条件或样本的候选仍可能走普通准入，详见下方状态表。

## 快速开始

```bash
# 环境：Python 3.10+；基础运行时无强制第三方依赖，测试和图像演示需要额外依赖
python -m pip install -e ".[dev,images]"

# 全部测试（252 项；耗时因环境而异）
PYTHONPATH=src python -m pytest -q

# 演示 1：经验记忆主线（离线，不调用任何模型，不花钱）
PYTHONPATH=src python scripts/demo.py

# 演示 2：依赖历史的视觉任务（离线）
PYTHONPATH=src python scripts/demo_history.py

# 演示 3：策略变更时旧证据怎么处理（离线）
PYTHONPATH=src python scripts/demo_policy_scope.py

# 知识库 RAG：导入文档、检索、带引用回答、评测（离线，不需要模型）
PYTHONPATH=src python -m evomemory.kb_cli demo --modes bm25

# 同一套流程，改用真实文本 embedding（需 pip install -e ".[rag]"）
export HF_HOME=$PWD/cache/huggingface/rag      # 模型缓存留在项目内
PYTHONPATH=src python -m evomemory.kb_cli demo --embedder fastembed --modes bm25,embedding,hybrid
```

三个演示都是**脚本策略**驱动，输出里明确标注 `harness_selfcheck / scripted`。
它们验证的是 harness 与任务设计，**不是模型效果**。

Windows PowerShell 中，在项目根目录执行以下命令；安装步骤与上面相同：

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
python scripts/demo.py
python scripts/demo_history.py
python scripts/demo_policy_scope.py
```

### 演示 1 会看到什么

8 个任务、同一 family。前 3 个任务遵循策略 A，从第 4 个起策略**悄悄换成 B**，family 名不变，
所以按 A 学到的经验仍会被检索到。两次运行唯一的差别是"允不允许退休失效的经验"：

```
WITHOUT evolution   2/8   显式跳过 14 次，引用式 strike 0，退休 0 条
WITH evolution      2/8   显式跳过  8 次，引用式 strike 2，退休 1 条

candidates: 36 record-level training samples replayed against
            accepted=1  rejected=5  pending=0
first rejection: replay:rejected ['action conflicts with observed evidence']
```

**这里要读懂两件事**：

1. 本演示中有结构化条件的第一个候选经过回放 `accepted` 后才被后续任务复用。
2. 策略变更后，编码新规则的候选被回放**拒绝了**——它与旧策略下已评分的记录冲突。
   这正是这道门该做的事，也解释了为什么两组最终都是 2/8：退掉失效的**规则**不够，
   它产生的**证据**仍然留在回放集里。按时间窗口或按证据退休属于后续工作。

退休本身仍然可查，演示会打印原因：

```
experience exp_8511…  task=escalation_03_118  record=r02
reason: agent explicitly skipped citing this experience; the experience
        exempted the record; grader required it
```

（本次运行的原始数字见 `artifacts/demo/<时间戳>/`；上面的计数以最新一次运行为准。）

### 演示 2 会看到什么

规则：某个区域**在先前面板里红过至少 2 次**就锁定，必须保持关闭。这个次数不在当前图片里、
不在任务文字里、任何工具也查不到——只能来自历史。

```
image-only   2/6      // 只看当前面板，遇到锁定区必然打开它
with history 6/6      // 用历史计数，全部正确
```

同一个演示还会打印"**像素完全相同、历史不同 → 正确动作不同**"，这是负迁移的最小例子。

历史**不是**从内存对象直接读的：每张面板的图像与区域事实都写入证据库（sha256 命名），
第 N 个面板的计数由回读第 0..N-1 个面板的证据重建，演示里会打印对应的 `evidence_id`。
但两个策略都是**确定性替身**，不是模型——这只证明任务有区分力，不证明模型会用视觉历史。

### 演示 3 会看到什么

同一批 8 个任务、同一台脚本策略，第 3 个任务起**判分规则变了**，最后 2 个任务是保留测试。
两个臂唯一的差别是"调用方有没有声明每个任务属于哪一次策略发布"：

```
不声明 release   2/8   回放：accepted, rejected, rejected, rejected
                       保留测试 0/2
声明 release     4/8   回放：pending, accepted, pending, accepted
                       活跃规则 2 条（旧 + 新）；保留测试 2/2
```

读法：不声明时，新规则与**旧策略时期评分过的记录**冲突，回放永远拒绝它；
声明之后，回放只取同一次发布的证据，新规则在攒够 2 个任务后被接受。

**这不是漂移检测。** 发布版本必须由外部给出（8 个任务例如 `--policy-scope A,A,A,B,B,B,B,B`）。
没有可信发布通知时，系统会保持保守——这是刻意的取舍，不是缺陷。

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
- 提供的远程实验命令使用同一份账本：`--llm api` 缺少 `--shared-stage3-budget` 会拒绝启动；
  视觉实验脚本也接入账本。离线开发用 `--llm scripted`。

## 知识库 RAG

```
文档导入 → 切分 → 索引 → 提问 → 召回 → 带引用回答 → 评测
```

```bash
# 五个子命令
PYTHONPATH=src python -m evomemory.kb_cli import --corpus data/kb_corpus
PYTHONPATH=src python -m evomemory.kb_cli search --query "灰度第一阶段放多少流量" --mode bm25
PYTHONPATH=src python -m evomemory.kb_cli ask    --query "回滚结束后需要留下什么记录"
PYTHONPATH=src python -m evomemory.kb_cli eval   --modes bm25,embedding,hybrid --embedder fastembed
PYTHONPATH=src python -m evomemory.kb_cli demo --embedder fastembed --modes bm25,embedding,hybrid
```

**三种检索，只有一种叫语义**：BM25、真实文本 embedding（`BAAI/bge-small-zh-v1.5`，
本地 ONNX，不依赖 torch）、以及两者 RRF 混合。另有词法哈希基线，它在所有输出里都被标成
`lexical`——不会冒充 embedding。

**固定问题集上的结果**（8 篇文档 / 12 个 chunk / 14 个问题，k=4，探索性）：

| 检索 | Hit@4 | Recall@4 | MRR |
|---|---|---|---|
| BM25 | 0.917 | 0.917 | 0.875 |
| embedding | 0.917 | 0.875 | 0.875 |
| hybrid | 0.917 | 0.917 | **0.917** |

三个通道都在同一道困难题（q11）上失败；在这个小语料上 embedding 没有跑赢 BM25。
引用有效率 1.000，引用命中率 0.917。细节与限制见 `docs/RAG_DESIGN.md`。

## 架构

| 模块 | 职责 |
|---|---|
| `harness.py` | Agent 循环、Memory Gate、token 预算上下文、Trace、Replay、Recovery、`check_rule` 可执行规则工具、归因式演化 |
| `memory.py` + `db.py` | 版本化经验库：不可变版本行、7 状态机、6 类写入者权限、追加式事件日志 |
| `predicate.py` | 三值谓词引擎（all_of / any_of / not / 比较符），深度与字段白名单，禁用 `eval` |
| `counterexample.py` | 候选经验在训练证据上的回放验证、结构化差异 |
| `lifecycle.py` | 经验提取（模型 + 确定性双路）、9 项校验、critique、重建 |
| `evidence.py` | 内容寻址证据存储（sha256），派生表示带 `derived_from` 血缘 |
| `kb.py` / `embeddings.py` / `kb_eval.py` | 文档知识库：导入切分、三路检索、带引用回答、固定问题集评测 |
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
| 经验生命周期（候选→验证→激活→修订/停用，旧版保留） | 已实现，252 项测试 |
| **结构化候选的回放** | 主线中有条件和样本的候选使用 `admit_with_replay`；未声明策略版本时，没有结构化条件或样本的候选仍可走普通准入，不能声称所有候选都经过回放 |
| **策略发布作用域**（旧证据不永久阻挡正确新规则） | 已实现并有对照演示；作用域由外部声明，非自动检测 |
| 测试任务隔离（不写证据、不动记忆） | 已实现，有测试 |
| 版本事务化切换、恢复一致性 | 已实现，有回归测试 |
| 三值谓词引擎 + 从反例推断连接词 | 已实现，39 项测试 |
| 历史依赖视觉任务（离线） | 已实现，12 项测试证明区分力 |
| 记忆对真实模型有提升 | **有观察**：0/18 → 61%（McNemar p=0.001）；未在更多任务上重复 |
| 显式跳过 + 经验引用在真实模型上可用 | **有观察**：23/23 次跳过都引用了经验（单次运行） |
| 历史任务对真实模型有区分力 | **有观察**：1 个跑完的锁定面板出现预期过度操作（n=1） |
| 完整生命周期优于简单记忆 | **未获支持**：11/18 vs 10/18（p=1.0），成本更高 |
| 结构化表示优于散文 | **未获支持**：6/8 vs 6/8（p=1.0），贵 43% |
| 视觉记忆带来增益 | **未证明**：原面板任务存在文字冲突或过于容易的风险，历史任务仍缺完整真实模型对照 |

## 已知边界

- **原面板任务的解释受限**：v2 文字与判分规则冲突；v3/v4 的已完成少量无记忆样本显示任务可能过于容易，但不足以证明总体天花板。后续演示改用 `history_bench.py`。
- **跨模态检索未实现**：当前是"文本索引 + 视觉证据复用"，不是向量级跨模态 RAG。
- **RAG 的真实模型回答未运行**：`ask --llm api` 已接线但本轮未调用；离线替身只会引用最佳片段，
  **无法判断"证据不足"**（实测失败，见 `docs/RAG_DESIGN.md` §3.2）。
- **知识库只支持 `.md` / `.txt`**，未做 PDF；经验辅助文档检索尚未接入。
- **策略变更需要外部声明**：`--policy-scope` 给定每次发布后，系统能只按该发布的证据回放并学到
  新规则；**不声明时旧证据会一直挡住新规则**。系统不会自己判断策略变了。
- **数据范围**：文本默认是逐任务在线学习；使用 `--holdout-from` 后的任务不写入训练证据或更新记忆。作用域对照的两个保留任务与训练任务来自同一生成器，不能证明跨分布泛化。
- **演示不展示同一经验的 v1 → v2 修订**：退休会生成新经验。原地修订（`revise` +
  `admit_with_replay`）已实现并有测试，但演示没有走到。
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
