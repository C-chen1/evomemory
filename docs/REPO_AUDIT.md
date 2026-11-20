# 三仓库源码审计（综合结论）

审计日期：2026-11-20
审计对象与冻结版本（commit 均为克隆时 HEAD，可按此复现）：

| 仓库 | 本地路径 | Commit | 许可证（实测） | 代码规模 |
|---|---|---|---|---|
| Recuris | `references/recuris` | `7d3745ab787b1206ccd981cb1511100476097266`（2026-08-30） | **Apache-2.0**（`LICENSE:1-3`） | 210 tracked / 72 `.py` / ~2.9 万行 |
| MemHarness | `references/memharness` | `74db61ebc4b64a23645f157cf440c66b6c975e5a` | **Apache-2.0**（`LICENSE:1-5`，README badge 误写 MIT） | ~491 `.py`，主体是 verl fork |
| CMA-Harness | `references/cma-harness` + `references/cma-harness-files` | `84732bcd6e838d1917d17184b98d3e416e4325c9`（单 commit） | **无许可证 = 保留所有权利** | 79 路径，`research/` 38 个文本文件 |

逐仓库的详细证据（文件行号级）在 `docs/_audit/recuris.md`、`docs/_audit/memharness.md`、`docs/_audit/cma_harness.md`，本文只做交叉结论与取舍。

> 关于 CMA-Harness 的获取方式：完整 `git clone` 因仓库内含 8 个 mp4 与多份 pdf、且本机到 GitHub 的连接不稳定，checkout 反复超时。处理方式是 `--filter=blob:none` 取 tree，再用 `git cat-file` 按 blob SHA 只抽取 `research/` 下的文本源码，共 38 个文件，原始路径清单见 `references/cma-harness-files/_BLOB_MANIFEST.txt`。工作区为空，`.git` 保留，SHA 可核对。**没有伪造任何未获取到的文件。**

---

## 0. 一句话结论

三个仓库**都在做"让 agent 从经验中学习"，但三者发布的代码都不能复现各自论文的头部数字**，而且失败原因各不相同：Recuris 是"机制在文档里有、在代码里没接线"；MemHarness 是"核心能力在发布配方里没被训练、对照组被污染"；CMA-Harness 是"数据与权重没发布、并且没有授权"。

这对 EvoMemory 的直接含义不是"照抄哪个"，而是：**这三份代码最大的价值是提供了一套可反驳的设计清单。** 我们的验收标准从第一天起就必须是"每个机制有一个能在 CI 里跑起来的断言"，否则会复刻同样的"看起来有机制、实际没运行"。

---

## 1. 核心问题：三者在同一个问题上切了不同的一刀

| | Recuris | MemHarness | CMA-Harness |
|---|---|---|---|
| 学习对象 | **记忆包** `M=(E,W,ρ,C)`，agent 权重冻结 | **策略权重**（GRPO），记忆是外挂上下文 | **VLM 权重**（SFT+RL），记忆是视觉片段库 |
| 学习信号 | 轨迹 → meta-agent 定位到某个分量 → patch → 配对 gate 准入 | episode return + 格式奖励 | 下游检索命中率（Jaccard） |
| 是否改模型 | 否 | 是 | 是 |
| 记忆形态 | Markdown/YAML 文本卡片 | 文本原则 + 1024 维向量 | 图片路径 + tags + 描述（无向量） |
| 抗负迁移手段 | 配对 held-out gate + reg_cap | critique 后 `<EMPTY>` 拒绝 | 结构性负样本 + 训练检索器 |

三者的"记忆"根本不是同一样东西。Recuris 的记忆是**可审计的规则库**，MemHarness 的记忆是**可被改写的提示片段**，CMA 的记忆是**可被索引的历史图片**。EvoMemory 要做的统一，是把这三者收敛到**同一个 Experience 记录**上，用 `evidence_refs` 同时指向文本、工具回执和原始图片。

---

## 2. 每一份代码里"说的"和"做的"差在哪（这是本次审计最有价值的部分）

### 2.1 Recuris：机制声明与实际接线的距离

代码里真正可复用的是一个约 2.5k 行的 kernel（`runtime.py` 447 行 + `wm/` + `em/` + `guards/` + `grounding.py` + `spi.py`）；而跑实验的脚手架极大——仅 `metaagent/driver.py` 就有 11058 行，`settle.py` 3274 行。**"机器很小，脚手架极大"** 这个比例本身就是一条设计教训。

明确查到的"声明未落地"：

- `WMSchema.field_writers` 只在 `wm/schema.py:88` 声明，全仓库无读取点；真实写权限硬编码在 `wm/ledger.py:162-201`。
- `gates.leakage_check` / `gates.fingerprint_verify` 只有定义（`gates.py:78-96`），主循环只 import 了 `held_out_paired_gate`。`gates.py:1-15` 自称 "four gates, all of them code"，实际只有 1 个被使用。
- `blocked`（do-not-repeat ledger）在 autonomous run 里恒为空：`blocked_additions` 永远是 `[]`（`driver.py:10396-10403`），`gates.Ledger` 全仓库无实例化。
- `configs/` 不被任何 runner 读取，README 说 "use the shipped configs" 在代码里没有加载路径。
- `skill_memories/skillflow/manifest.yaml` 引用了 `builtin` 里不存在的 checker 和 cfg key，`load_skill_memory` 会抛错——即 SkillFlow 这条线**根本不经过 kernel**。
- 仓库**没有 `tests/`**，而多处注释声称 "byte-parity tests keep guarding them"（`builtin/entrykinds.py:1-5`、`wm/render.py:78`）。测试没有随仓库发布。

另有一处对结论影响最大的发现：campaign 的 subprocess 环境过滤掉所有 `TAU2_*`（`downstream.py:622, 679-695`），之后只补 `TAU2_DATA_DIR`。因此 `metaagent run` 的每次评估里，`TAU2_GATE_TERM` / `TAU2_STATUS_BOARD` 都是关闭的，尽管包的 manifest 写着 `board.status_board: true`。**演化循环实际测到的不是 README 描述的那套 harness 机制。**

这批发现的说服力在于它们是**可复现的静态事实**（grep 就能验证），不需要跑实验。

### 2.2 MemHarness：核心能力的训练证据不在发布物里

README 主张"critique + reconstruction 能力通过 GRPO 端到端涌现"。代码事实：

- `agent_system/` 下搜索 `critique` / `reconstruct` **0 处命中**（只在 README 出现）。实际实现是**一次额外的 LLM 改写 pass**：`mem_adaptor_rollout.py:68-101` 判定，`:431-454` 拼输出，`:512-520` 拒绝时删块。
- critique 的判定信号只有字面量 `<EMPTY>`，没有任何数值阈值参与；`state_embedding.py` 算的 S_old/S_curr 余弦**只写日志**（`mem_adaptor_rollout.py:706-720`）。
- 决定性证据在发布脚本 `run_scripts/train_alfworld.sh:255-258`：`mem_adaptor.enable=True` 但 `mem_adaptor.train_memory_adaptor=false`。而 `train_memory_adaptor_enabled()` 要求 `use_actor_rollout_wg=false ∧ train_memory_adaptor=true`（`mem_adaptor_training.py:331-338`）。因此改写 token **不进 GRPO 损失**。
- 仓库里确实实现了针对 adaptor 的第二套 GRPO（`mem_adaptor_training.py:387-467`），但发布配方没启用。
- 100% 纯 GRPO 对照被污染：`FORMAT_REWARD_CLI` 无条件 `require_memory_retrieve=True`（`train_alfworld.sh:188-196`），关掉记忆后格式奖励仍要求 `<memory_retrieve>`。
- 语义去重在训练脚本里被显式置 null 关闭（`train_alfworld.sh:135-136`），而 README 把 "semantic deduplication" 列为特性。
- 返回的失败轨迹也会入库（`only_store_positive_reward` 默认 False，`ppo_trainer.yaml:415`），与 README "successful trajectory" 的叙事不符。

**这对我们的意义很具体**：发布配方下的 critique/reconstruction **本质上是零样本 prompt 行为，不需要 GPU 也能复现其结构**。没有 GPU 损失的是 (a)"改写能力由训练涌现"这一因果主张，(b) latent guidance（关记忆仍 83.0%）——那必须靠 checkpoint。所以 EvoMemory 复现的是**结构与生命周期**，不声称复现 +8.8/+9.5。

### 2.3 CMA-Harness：授权问题比技术问题更硬

- 全仓库**没有任何 LICENSE / COPYING**。唯一相关的 `research/THIRD_PARTY_NOTICES.md` 末句逐字为：*"This file does not assign a new license to the authors' research code."* 这是在**反向澄清**：Apache-2.0 只覆盖 `third_party/ms-swift-LICENSE` 对应的 ms-swift，不覆盖 `research/` 下任何 `.py` / `.sh` / prompt。
- 结论：**不能把 CMA 的代码搬进 EvoMemory**（无论是否注明出处）。可迁移的只有思想——算法、架构、数据模型、指标定义不受著作权保护。
- 代码完整度：**不是"coming soon"**。`research/README.md:14-20` 明确说这是 exported research snapshot，"does not contain its complete 17-tool integration"，且 "GPU training, full inference and paper-score reproduction have not been rerun for this export"。已发布推理服务主干（960 行 `multimodal_agent_service.py` + 270 行 FastAPI），但**权重、M2CA-Bench、训练数据全缺**（`.gitignore:4-6` 排除了 `checkpoints/`、`datasets/`、`scenarios/`）。
- 技术层面最反直觉的一点：所谓"记忆压缩"只有 448 缩略图 / tags / desc 三种**定性消融**，**没有 embedding、没有向量索引、没有遗忘合并**，检索是每轮对全量记忆做一次线性扫描；代码里**没有任何 token 成本量化**，而 README 声称 "reducing visual-token overhead"。
- 另有一处训练-推理不一致可以直接做实验：写入端要求 description ≥100 词，检索端截断到 300 字符（`multimodal_agent_service.py:591`）——**一半以上的描述永远读不到**。

---

## 3. 许可证矩阵（决定哪些代码能进 EvoMemory）

| 来源 | 许可证 | 能否复用代码 | 义务 |
|---|---|---|---|
| Recuris 自身代码 | Apache-2.0 | **可以** | 保留 LICENSE + NOTICE，标注修改 |
| Recuris vendored：tau2 | MIT | 可以（但仓库选择 fetch+patch 而非分发） | 保留版权声明 |
| Recuris vendored：harbor | Apache-2.0 | 可以 | 保留声明 |
| Recuris vendored：tb21 | Apache-2.0 | 可以（镜像/任务数据另有版权） | 保留声明 |
| **SkillFlow**（Recuris 的 third_party 之一） | **无许可证** | **不可以** | —— |
| MemHarness 自身代码 | Apache-2.0（badge 误写 MIT） | **可以** | 保留声明、标注修改、**说明该 badge/LICENSE 不一致** |
| MemHarness 依赖：verl / verl-agent | Apache-2.0 | 可以 | 保留声明 |
| **CMA-Harness 全部 research 代码** | **无许可证，明确不授权** | **不可以** | 只能独立重实现思想 |

EvoMemory 的实际做法：**一行代码都不复制**。三个仓库只作为设计与反例来源，全部模块自己写。这样既避开许可证不确定性，也避免继承"声明未接线"的历史包袱。若日后确实要复用 Recuris 的 kernel，再单独按 Apache-2.0 走 NOTICE 流程。

---

## 4. 十个维度的横向对比

### 4.1 关键算法

| 维度 | Recuris | MemHarness | CMA-Harness |
|---|---|---|---|
| 检索 | 事件驱动 deliverer（`turn_start` / `intent_recorded` / `pre_write`），可选 embedding | 余弦 top-k=3，**无有效阈值**（`min_retrieval_score=0.0`） | **无向量**，VLM 读全量缩略图输出 rank |
| 归因 | LLM 提议分量 + 确定性校验（`plan_schema.validate_plan`） | 无归因 | 无归因 |
| 改写 | meta-agent 直接改包文件 | 一次 LM pass，`<EMPTY>` 拒绝 | 无 |
| 准入 | 配对 bootstrap，`accept = lo>0 ∧ n_dn<=reg_cap` | 无 | 无 |

Recuris 的 `held_out_paired_gate`（`gates.py:31-75`）是三者中**唯一**真正的准入机制：对 **item（task）** 重采样 3000 次而非对 trial，取 2.5/97.5 分位，`accept = lo > 0 and n_dn <= reg_cap`。这个"重采样 item 而不是 trial"的细节很重要——对 trial 重采样会给出伪窄置信区间。

### 4.2 数据模型

- Recuris：`EMEntry(id/type/body/trigger_event/trigger_tool/source/path/extra)`，`type ∈ {knowledge, procedure, action_result}`。**没有媒体字段、没有 embedding、没有 superseded/deprecated 状态**。包级版本是目录不可变（`versions/M{rnd}`），不是卡片级版本。淘汰一张卡只能 `remove_card`——TTA 的 prompt 说 "you may supersede it"，实际实现只是再写一张新 id 的卡，旧卡仍在。
- MemHarness：`MemoryRecord(...)` + `value=(succ+1)/(use+2)`，**无模态字段**。
- CMA：`{id, rank, tags[], title, description, diff, image_path}`，**无 parent_id、无 embedding、无 source、无 turn/topic**。编辑链只能靠自然语言 `diff` 猜。

三者的共同缺口正是 EvoMemory 的 `Experience` 要补的：**`evidence_refs` + `memory_version` + `verification_status` + `supersedes` 显式血缘**。CMA 缺血缘导致无法做版本树；Recuris 缺卡片级版本导致无法回答"这条经验被改过几次"。

### 4.3 执行生命周期

Recuris 的 kernel 生命周期（`runtime.py:192-266`）值得直接照搬结构：`_ground`（用真实回执结算上一轮证据）→ `turn_start` deliverers → WM manager 判断是否更新 → `render_wm` 重编译上下文 → `llm_draft` → `draft_ready` checkers（最多一次 redraft）→ `commit` → `pre_write` deliverers → sanitizer。注意它与文档顺序相反：**先结算上轮、再决定本轮**。

### 4.4 记忆存储

- Recuris：**纯文件**（`em/<type>/*.md`），无 SQLite、无向量库；embedding 缓存用 `sha1(embedder + "|" + sorted(doc_ids))` 命名 `.npy`——**key 只含 id 不含 body，`edit_card` 保留 id 时旧向量不会失效**，且 `np.save` 非原子写。
- MemHarness：Milvus Lite + FLAT/COSINE，每任务一个 collection，`update_records` 是 delete+reinsert，且**检索时对每条命中同步 update**（等于读放大成写）。
- CMA：图片路径 + `/tmp` 缩略图缓存，缓存键 `md5(image_path)[:12]` **不含 mtime/size**，文件被覆盖会命中旧缓存。

三者都栽在同一个地方：**内容寻址不彻底**。EvoMemory 的做法是证据文件全部按 `sha256(content)` 寻址、经验按 `version` 递增且不可变。

### 4.5 经验更新机制

只有 Recuris 有真正的"经验更新"闭环，且它的接受条件是**算术**而不是模型投票（"No model votes on its own patch"）。MemHarness 的 utility 是关联计数（episode 成功就给该轨迹所有被检索过的 id `c_succ+=1`），不是因果归因。CMA 完全没有更新——记忆只增不减。

### 4.6 失败处理

最能体现工程成熟度的是 Recuris 把最贵的 bug 写成了结构性防线：模型不能写 DONE（`wm/ledger.py:162-168`）、synthetic id 注册后再 ground 一律拒绝（`runtime.py:271-275`）、unmatched receipt 只计数不伪造 entry（`grounding.py:187-194`）。

但它的 TB2.1 分支明显更弱：`runtime_bridge.py:309-319` 在 checker 不 bounce 时把**所有 pending entry 一次性 mark_done**；没有回执时只记一个计数仍然放行（`:320-322`）。同一仓库里同一机制有强口径和弱口径两种实现——这说明"grounding"必须由**单一实现**提供，不能每个 adapter 各写一套。

MemHarness 是"读路径硬失败、写路径软失败"：retrieve 抛异常不捕获，直接穿到 rollout 失败。

### 4.7 Benchmark 设计

三者的共同做法值得抄：**同一函数生成两个 arm、只切换一个变量**。Recuris 的 `render_one` 用同一个函数产出 bare/skill，两者只差 `prompt_template_path`；`--routing default` 不使用任何看过分数的 per-family 选择，`frozen_insample` 明确标注 in-sample 并在 stderr 警告。

三者的共同问题是**预算不等**：Recuris 的 skill arm 每轮多一次 ledger update、bounce 时多一次 redraft，所以"同预算"只成立于 task×trial 数，不成立于 token 数。EvoMemory 必须同时对齐 **task 数、trial 数、token 预算、tool-call 上限**，并对超出预算的 run 显式标记。

### 4.8 可复用组件（接口级，不是代码级）

按"低耦合、值得自己重写一遍"排序：

1. `HarnessPort` 这组约 12 个方法（`runtime.py:47-62`）——把任意 agent 循环接到 kernel 的唯一缝。
2. 写权限三通道（`model` / `harness` / `oracle`）+ **evidence admission 与 matching 分离**（`grounding.py:137-195`）：matcher 只能决定"哪条 entry"，不能决定"是否准入"。
3. 六个 SPI Protocol（`spi.py:47-210`）：`EntryKind` / `WMManager` / `Deliverer` / `GroundingMatcher` / `Checker` / `FeasibilityOracle`。
4. `held_out_paired_gate` + power gate（`gates.py:31-75`、`driver.py:9975-10029`）：开跑前用 MDE 拒绝欠功效设计。**5-10 个任务上的单轮提升根本不可检测**，这一条对 EvoMemory 尤其重要。
5. `activation_probe`（`activation_probe.py:361-433`）：离线跑真实 selector，证明"这张卡会被投递"。**先证明记忆会触发，再测效果。**
6. 机制指纹先于分数（`fingerprint.py`、`scorecard.py:197-250`）：scorecard 拒绝在机制计数之前打印分数。
7. MemHarness 的 `MemoryEvent`（每步 `query/state/injected/retrieved`）与"结构化记忆块可定位/可替换/可删除"。
8. MemHarness 的 `retrieved_state_mode = matched|placeholder|random_vdb`：现成的反事实开关，直接回答"改写是否真的用了源状态"。
9. CMA 的"用下游检索命中率监督记忆写入器"：把记忆质量定义为"未来能不能被用上"，而不是 BLEU。
10. CMA 的"按会话切分 + 落盘 split meta"防泄漏。

### 4.9 改造难点

- Recuris：`metaagent/driver.py` 与 tau2 / Claude Code CLI / Anthropic proxy / 固定 split / 固定 task id / doubao 模型 / Docker+harbor 焊死。**不要移植 Driver**，重写一个 300-500 行的 round loop。
- MemHarness：`actor_rollout_wg.generate_sequences` 是 vLLM+Ray+GPU；`_summarize_batch_teacher` 这条路径本来就是 OpenAI 兼容 API，可直接换。
- CMA：8 卡角色硬编码（A=0, B=1, C=2, Qwen-Image=3-7），`gpu_count == 0` 直接 `RuntimeError('No available GPU; model cannot run on CPU')`。

### 4.10 明确的"无 GPU / 无 API 不可复现"清单

| 事项 | 原因 |
|---|---|
| Recuris 的 results 表（35/37 对、+23.3） | 仓库里没有 runs/artifacts，`git ls-files` 只有 210 个文件 |
| Recuris 的演化 campaign | 需要 Docker + Compose V2 + Claude Code CLI + 在线 judge |
| MemHarness 的 85.2/75.6 与 latent guidance 83.0 | 需要 8 GPU + checkpoint；且发布配方未训练 adaptor |
| MemHarness 的 OOD 85.9 | 同上 |
| CMA 的任何 benchmark 分数 | 需要 7-8 张 40GB 卡 + 未发布的权重与 M2CA-Bench |
| CMA 的 VLM A/B RL | 8 卡 + ZeRO-2 + flash_attn + 独立奖励服务进程 |

能在 CPU 上真实执行的：CMA 的 14 个 CPU 单测（`research/tests/test_release.py`），以及 Recuris 的 `gates.py` / `integrity.py` 这类纯函数——**后者正是我们验证自己 gate 实现的黄金参照**。

---

## 5. 三个仓库共通的三条教训

1. **"文档声称"是高危区。** 三个仓库都出现了 README 与代码不一致，而且都不是小差异：Recuris 的 harness 机制在演化时被关掉、MemHarness 的 adaptor 不进损失、CMA 的"记忆压缩"没有任何量化。EvoMemory 的对策是**每个机制配一个能跑的断言**，并且 `docs` 里每个"已实现"标记必须能指向一个测试名。

2. **"同预算对照"比想象中难。** Recuris 的 bare arm 没有 harness hook，MemHarness 的 baseline 被格式奖励污染。EvoMemory 的 A/B/C/D 必须由**同一个 runner** 驱动，只通过一个 `mode` 参数切换，并在结果里记录每个 run 的实际 token/tool 调用数。

3. **记忆的价值必须在"会被用上"之后才谈效果。** Recuris 的 `activation_probe` 和 MemHarness 的 accept-rate 都指向同一件事：先证明检索/投递真的发生了，再谈成功率。否则测出来的是噪声。

---

## 6. 对 EvoMemory 的直接影响（写进设计的硬约束）

1. **不复制任何一行第三方代码**，全部独立实现；设计灵感在 `docs/INTEGRATION_DESIGN.md` 中逐条标注来源（arXiv 编号）。
2. **记忆必须卡片级版本化**（三个仓库都缺），且 `supersedes` 显式成边，可回滚。
3. **证据必须内容寻址**（`sha256`），原始图片永不覆盖、永不 resize 回写——CMA 的 `align_image_to_model` 原地覆盖图片是反面教材。
4. **准入用配对 bootstrap + reg_cap**，并先算 MDE；任务数不足时明确报"欠功效"而不是报一个好看的数字。
5. **每个机制一个 CI 断言**：写权限、grounding、gate、版本、检索、预算，逐项有测试。
6. **多模态不做未经验证的效果声明**：多模态模块的存储/检索/压缩是可测的；"视觉是否提升成功率"标记为待测，必须等真实 VLM 接入。
