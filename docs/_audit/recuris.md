# Recuris 源码审计

审计对象：`D:\python\PyCharm\Project\06_evomemory\references\recuris`（git `7d3745a`，210 个 tracked 文件，72 个 `.py`，约 2.9 万行）。
方法：只读阅读 README/docs 与源码，未运行、未安装、未修改任何文件。
标记：**【代码】**= 直接读到实现；**【文档】**= README/docs 的声明；**【推断】**= 我根据代码结构做的判断，未运行验证。

仓库体量分布很说明问题：核心 kernel（`runtime.py` 447 行 + `wm/` + `em/` + `guards/` + `grounding.py` + `spi.py`）约 2.5k 行；而 `metaagent/` 下仅 `driver.py` 就 11058 行、`settle.py` 3274 行、`downstream.py` 1814 行。也就是说，"可复用的机器"很小，"跑实验的脚手架 + 冻结协议"极大。

---

## 1. 核心问题与主张

**【文档】**README 第 37–43 行把主张说得很清楚：不改权重、不改 prompt，只改一个冻结 agent 外挂的 **Skill Memory `M = (E, W, ρ, C)`**；meta-agent 读结构化轨迹，把失败定位到 M 的某一个分量，只 patch 那一个分量；一个确定性的 gate 在配对 held-out 证据上决定是否保留。第 45–54 行补充四个特性：状态驱动检索、分量级定向演化、validation gate、training-free/model-agnostic。

**【代码】**这套主张的落地位置：

| 分量 | 代码中的实体 | 证据 |
|---|---|---|
| `E` 经验记忆 | `EMEntry`（frontmatter + markdown body）+ `EMStore` 目录加载 | `src/recuris/em/entry.py:21-56`、`src/recuris/em/store.py:16-27` |
| `W` 工作记忆 | `WMSchema` / `Ledger` / `EntryKind` / `WMManager` / `render_wm` | `src/recuris/wm/schema.py:75-97`、`wm/ledger.py:66-205`、`runtime.py:223` |
| `ρ` 调用策略 | manifest `delivery:` 列表 + deliverer 的 `events`/`at` + 卡片的 `trigger` | `skillmemory.py:211`、`runtime.py:135-154`、`builtin/deliverers.py` |
| `C` checkers | manifest `checkers:` + `TruthProtocol`/`ExecutionGate`/`AntiEscalationGate` + `local:plugin` | `builtin/__init__.py:46-50`、`guards/truth.py:75-135`、`builtin/execution_gates.py:65-160` |

manifest 里还有 `board`、`gate`、`feasibility` 三个包级策略，不在四元组里，但会影响行为：状态板（`board.py`）、终端门读取的 pending 行（`runtime.py:165-182`）、airline 的 feasibility oracle（`skill_memories/tau2_airline/plugin.py`，由 `runtime.py:421-431` 以 `Writer.ORACLE` 写 BLOCKED）。所以论文的 M 是概念抽象，实际包格式比四元组多三项。

"frozen agent" 的准确含义是冻结权重和 agent policy，不是冻结输入：每轮把 `render_wm()` 拼到 system prompt（`adapters/tau2/agent.py:189-198`），checker bounce 还会追加 correction 再生成一次（`runtime.py:251`、`agent.py:200-208`）。因此更准确的说法是"冻结模型 + 可变上下文"；这一点 README 的 "frozen agent" 容易读过头。

另有一个重要边界：**三个 benchmark 并没有跑同一台机器**。tau2 跑完整 `TurnRuntime`；TB2.1 用 `TerminalSkillBridge` 复用 `TurnRuntime` 的 ledger/deliverer/checker，但自己实现 completion 与 retrieval 事件（`adapters/tb21/runtime_bridge.py:52-323`）；SkillFlow 完全没有 kernel，只在渲染模板里注入 `MACHINE.md`/cards（`adapters/skillflow/render_configs.py:51-110`、`skill_memories/skillflow/manifest.yaml:1-17`）。【推断】README 的 "same machine across arms" 在单个 benchmark 内成立，跨 benchmark 不成立。

---

## 2. 关键算法

### 2.1 回合算法（不是论文里的抽象顺序）

`architecture.md:33-42` 写的是 classify → update → deliver → draft → check → ground → commit。**【代码】**实际 `TurnRuntime.run_turn`（`runtime.py:192-266`）是：① `_ground` 先用真实 receipt 结算上一轮证据（`204`、`269-280`）；② `turn_start` deliverers（`209`）；③ WM manager 判断是否更新 → `_update_ledger` → `intent_recorded` deliverers（`212-220`）；④ `render_wm` 重编译上下文（`223`）；⑤ `llm_draft`（`228`）；⑥ `draft_ready` checkers，最多一次 redraft（`231-252`）；⑦ `commit`（`255`）；⑧ `pre_write` deliverers / write-review bounce（`258`、`363-394`）；⑨ sanitizer 剥离模型伪造的 trusted block（`261-264`）。

所以"先 grounding 再更新"与文档顺序相反；文档是概念图，代码是先结算上轮、再决定本轮。

### 2.2 失败定位

**【代码】不是分类器，而是"LLM 提议 + 确定性约束"**：

- 每个失败任务的全部 k 条 trial 被写成 `evidence/task_<id>.md`，带失败 reward、termination reason、`action_opportunity_O`、transcript head/tail（`metaagent/driver.py:6617-6687`）。
- 诊断 prompt 要求按"失败步骤"聚类，而不是按任务故事；每个 cluster 必须填 `component ∈ {E,W,RHO,C,MODEL,HARNESS,BENCHMARK,CEILING}`、`action_opportunity ∈ {present,absent,unknown}`、`patchability ∈ {memory,external,unknown}`（`metaagent/protocols/diagnosis.md:48-58`、`plan_schema.py:25-32`、`91-219`）。
- 只有 `O=present 且 P=memory` 的 E/W/RHO/C cluster 才允许带 fixes；否则必须 `fixes: []`，即"弃权"是合法结果（`diagnosis.md:17-24`、`plan_schema.py:118-135`）。
- 确定性部分只有三处：`_action_opportunity` 从 evaluator 的 required write actions 与实际 tool-call trace 判断 O（`downstream.py:274-305`）；`reachability.py` 用 AST 解析 benchmark 的 tool capability 与 deliverer dispatch 事件（`reachability.py:139-208`）；`validate_plan` 校验 component/action/事件/tool/cfg 合法性（`plan_schema.py:231-790`）。

**【推断】**因此"failure localization 定位到某个分量"本质是上游模型的判断，代码只保证它不能写出一个与分量不匹配、载体不可达或引用不存在证据的 plan。这与 README 的 "locates each failure in one component" 有语气差别：不是自动定位，而是受约束的人工/模型诊断。

### 2.3 patch 生成

**【代码】**`phase_d` 产出 `plan.json`；`phase_p` 让外部 coding agent 在**一次性候选目录**里按 plan 编辑（`driver.py:7412-7756`、`8268-8505`）。`remove_card` 由 driver 先确定性删除（`8033-8046`），之后 `_plan_conformance_errors` 逐条比对 base/candidate 的文件 hash 与 manifest，要求"每个 fix 都实现且没有计划外修改"（`8048-8230`）。随后还有 lint、reachability、leak scan、activation probe、repair eval。

patch 的"生成者"不在仓库里：`cc_session` 调 `launchers/run_claude_code.sh`，后者调 `claude` CLI；模型请求经 `anthropic_openai_proxy.py` 转成 OpenAI-compatible（`driver.py:5584-5732`、`launchers/run_claude_code.sh:1-77`、`proxy/anthropic_openai_proxy.py:1-20`）。这是最重的外部依赖之一。

### 2.4 validation gate 的具体算术

核心在 `metaagent/gates.py:31-75`，`driver.py:9420-9675` 调用：

- 输入是 `base`/`cand` 两个 `dict[task_id -> [per-seed score]]`；
- 每个 task 先对 seed 取均值，`d_i = mean(cand_i) - mean(base_i)`；
- `net = mean(d_i)`；
- **对 item（task）重采样** 3000 次，取 2.5%/97.5% 分位；`accept = lo > 0 and n_dn <= reg_cap`（`gates.py:55-68`）；
- dense reward 时 `material` 阈值生效：`d_i > material` 才算 improved，`d_i < -material` 才算 regressed（`gates.py:48-62`），driver 默认 `DENSE_MATERIAL_DELTA=0.05`（`driver.py:199`）；
- `n_boot=3000, seed=0` 固定；`reg_cap` 是硬拒绝条件，默认 CLI 1（`driver.py:10960`）。

"配对"还体现在 arm 级约束：每轮 baseline 与 candidate **同时重跑**，并要求 `_require_paired_seed_maps`、同一 benchmark git commit、同一 benchmark input sha256、同一 package tree sha256（`driver.py:9490-9518`、`9363-9379`）。历史分数不被当作因果 baseline（注释 `9490-9492`）。

`--round-gate` 有四种实际语义（`driver.py:9568-9586`）：

- `relaxed`（**CLI 默认**）：`repair_ok and net>0 and n_dn<=reg_cap`；
- `strict`：`held_out_paired_gate.accept and repair_ok`；
- `calibrated/progressive`：`repair_ok and (p_rep<0.05 or p_dev_pos<0.05) and p_harm>=0.05`；
- `always`：无条件接受，用来测量 gate 本身的价值。

注意 `p_harm >= 0.05` 是"没有显著伤害"，不是"证明无伤害"；在 12 个 dev task 上这是一个很弱的 veto。

repair 先于 held-out：`rc > rb + 1e-9`，否则直接 REJECT，不花 held-out 预算（`driver.py:9469-9488`）。p 值用 `_mcnemar_one_sided`（二值）或 task-clustered paired bootstrap（dense，`driver.py:230-337`）。

**power gate**（`driver.py:9975-10029`）：

- 二值：对 cell 预算 `dev_tasks × k`，枚举 `regressions c ∈ {0,1,2}`，找最小 `b` 使 `P(X>=b | b+c, 1/2) < 0.05`，`MDE_pp = 100*(b-c)/cells`；若 clean MDE > `--power-gate-mde-cap`（默认 12pp）则在 `enforce` 下拒绝开跑，`warn` 下只警告（`driver.py:10016-10029`）。
- dense：没有闭式 MDE，改为模拟：假设 per-trial sd=0.15，合成 tasks×k 的配对差异，用真实 estimator 反推 80% power 的最小 effect（`driver.py:340-389`、`9855-9973`）。同时硬性要求 dev task 数 ≥5（否则 bootstrap 无法认证任何效应）且 k≥2（否则 false-accept 约 9.3%）；这些是**实际 enforce** 的，模拟 MDE 只报告不 enforce（`driver.py:9934-9972`）。

`_perfect_stratum_damage` 只做诊断：在 base 全对的 task 上数丢失 cell，并用 base 自身半 seed 选择/另一半读数估计 null，明确 `reject_authority=False`（`driver.py:452-520`）。

### 2.5 其他 gate

- `metaagent/gate.py` 是一个独立 CLI（`recuris metaagent gate`），用 10000 次 item bootstrap + repair + reg_cap，**与 driver 的 gate 实现重复**，driver 不使用它（`gate.py:30-75`）。
- `settle.py` 是"冻结候选的最终 A/B"：固定 `V2_FIXED_TASKS`（repair task 38，guard 50/33/36/86）、固定 trial seeds、每 arm 20 个逻辑 cell，接受条件 = 两个 repair block 都过 net≥2 且机制因果 cell 触发 + safety 新失败≤0 + 其他 guard 新失败≤1 + overall net>0（`settle.py:51-96`、`2071-2215`）。它不复用 driver 的 round gate，而是另一套冻结算术；20 cells 是"最终裁决"而非功效设计，换 benchmark 不能照搬。

---

## 3. 数据模型

### 3.1 包与 manifest

**【代码】**包 = 目录 + `manifest.yaml`（`skillmemory.py:167-223`），`SkillMemory` dataclass 字段：`name/root/wm_schema/manager/matcher_name/em/deliverers/checkers/status_board/board_notice/gate_lines/stay_notice/oracle/raw`（`skillmemory.py:26-42`）。`wm.schema` 支持 `binding_key/collection_key/max_entries/machine_id_pattern/board_marker`（`_build_schema`，`skillmemory.py:95-107`）。`WMSchema`（`wm/schema.py:75-97`）字段：`entry_noun`、`max_entries`、`MatchPolicy`（binding/collection/各 bonus/min_score）、`RenderPolicy`（machine_id_pattern/free_text_label/empty_label）、`board_marker`、`entry_kind`、`field_writers`。

**可疑**：`field_writers` 只有声明，全仓库无读取点（grep 仅 `schema.py:88`），而 docstring 说"ledger engine consults this before every write"。实际权限是硬编码在 `mark_done/mark_blocked/mark_authorized` 的 writer 检查里（`wm/ledger.py:162-201`）。

### 3.2 entry

`LedgerEntry`（`wm/ledger.py:27-63`）：`id/description/state/evidence/blocked_reason/fields/auth`；`to_dict()` 把 `fields` 摊平。`EntryState` 四态：`not_yet/done/blocked/obsolete`（`schema.py:23-33`）。`Writer` 三值：`model/harness/oracle`（`schema.py:15-20`）。

`EMEntry`（`em/entry.py:21-30`）：`id/type/body/trigger_event/trigger_tool/source/path/extra`；`type ∈ {knowledge, procedure, action_result}`；frontmatter 额外键进 `extra`。没有媒体字段、没有 embedding、没有 supersedes/deprecated 状态。

### 3.3 版本与来源

**【代码】**`Driver` 的 `state.json` 默认键：`round_done/best/best_dev/best_dev_seeds/best_dev_benchmark_git_commit/best_dev_benchmark_input_sha256/bare_dev_reference/sims/blocked/changelog/train_history/version_hashes/round_outcomes`（`driver.py:1858-1866`）。每轮接受后把候选复制到 `versions/M{rnd}`（`driver.py:9677-9750`），记录 tree sha256；`changelog` 只存 round/summary/net_pp；每轮另有 `ledger` event 文件、`round_r_lesson.txt`、`round_review.md`、`provenance.json`（含 config、runtime source hashes、evaluations）。

证据层面：每个 evaluation 的 `artifact` 记录 tag、task ids、num_trials、package_tree_sha256、agent_implementation、worker/simulator model+llm_args、benchmark git commit、benchmark input tree hash、trial seeds、results/fingerprint sha256（`downstream.py:623-670`、`776-787`）。这是整份代码里最值得借鉴的 provenance 设计。

---

## 4. 执行生命周期

### 4.1 单次 tau2 arm

`recuris` CLI（`cli.py:230-233`）→ `adapters/tau2/run.py:main`（`114-271`）→ 校验 treatment（`treatment.py:101-160`）→ `registry.register_agent_factory`（`run.py:184`）→ tau2 `run_domain(TextRunConfig)`（`run.py:194-213`）。tau2 每轮调 `RecurisAgent.generate_next_message`（`agent.py:286-295`）→ `TurnRuntime.run_turn` → `_Tau2Port` 的 `ledger_llm/llm_draft/bounce_and_redraft/commit`（`agent.py:175-244`）→ 返回最终 AssistantMessage。持久化：tau2 `results.json` + `RECURIS_FINGERPRINT` jsonl（`agent.py:307-317`）+ `_params.json`（`run.py:217-240`）。

### 4.2 演化 campaign

`recuris metaagent run` → `driver.main`（`driver.py:10847`）→ `Driver.run`（`10789`）：

```
init/integrity -> neutral()/bootstrap()/initialize_from_package()
for rnd in start..rounds:
    round(rnd):
      evaluate(working_base, train_ids, k)            # driver.py:10192-10194
      write_evidence -> evidence.md + per-task trial blocks
      phase_d -> plan.json (external session)
      _prepare_candidate + _apply_planned_deletions
      phase_p -> edits candidate (external session)
      lint/leak/activation/conformance
      gate: repair eval + contemporaneous dev eval + held_out_paired_gate
      commit -> versions/M{rnd} + ledger + changelog + lesson
      or reject -> discard/provisional
```

`evaluate` 本身不跑 in-process：它 subprocess 调 `recuris.adapters.tau2.run`（`driver.py:5898-5938`、`downstream.py:702-738`），所以每个 evaluation 是独立进程，崩溃/挂起不会带走 campaign。driver 用 `_reuse_initialization_eval` 按完整身份复用已封存 artifact（`driver.py:5940-6027`）。

TB2.1 是另一条链：`recuris tta run` → `tta/driver.py:main` → 线程池 `run_task` → `harbor run -c` → `RecurisTerminus2` 的 `_handle_llm_interaction`/`_execute_commands`（`harbor_terminus.py:52-95`）→ `TerminalSkillBridge.prepare_prompt/observe_commands/check_completion`。SkillFlow 没有 kernel：`render-configs` 生成 YAML，`harbor run` 执行外部 QwenCode agent，`score` 只读 harbor 的 `result.json`。

---

## 5. 记忆存储策略

**【代码】**全部是文件，没有 SQLite/向量库：

- E：`em/<type>/*.md`，一个文件一张卡；`EMStore.load` 只读 `*.md`、跳过 README（`em/store.py:16-27`）。W：内存对象 `Ledger`，不落盘；每轮 `RECURIS_WM_DEBUG` 可追加 JSONL 快照（`runtime.py:438-447`）。ρ/C：`manifest.yaml` 的 `delivery`/`checkers` 列表。
- 运行产物：`results.json`、`*.fp.jsonl`、`_params.json`、`provenance.json`、`plan.json`、`evidence/`、`versions/M*/`、`changelog`、`lesson`、`ledger` event 文件。完整性：`champions.lock.json` 存每个受保护包每文件 sha256+bytes，`integrity/anchors.json` 存聚合 sha256；`verify()` 逐文件比对 added/deleted/modified（`integrity.py:221-254`）。

**去重与指纹**：

- 卡片 id 唯一（`EMEntry.id` 默认文件名），交付去重键是 `(doc_id, entry.description)`（`runtime.py:325-327`、`builtin/deliverers.py:157-159`），因为 ledger entry id 每轮会 re-key。
- 包指纹：`_package_tree_sha256` 对 "相对路径:sha256" 排序后再 sha256（`downstream.py:87-118`）；`tree_sha256` 在 integrity 里对 "pkg/rel:sha256" 排序再 sha256（`integrity.py:71-76`）。
- 机制指纹：`Fingerprint` 每对话一个 `Counter` + `events` 列表，`event()` 既计数又追加结构化事件（`fingerprint.py:21-37`），`dump()` 追加一行 JSONL（`40-54`），`aggregate()` 汇总（`57-69`）。scorecard 强制先打印机制计数再打印分数（`tools/scorecard.py:197-250`）。

**embedding 缓存**：`EmbeddingRetrieval._ensure_index` 以 `sha1(embedder + "|" + sorted(doc_ids))[:16]` 作为 `.npy` 文件名（`builtin/embed_retrieval.py:89-103`）。**注意**：key 只含 id 不含 body，`edit_card` 保留 id 时旧向量不会失效；`np.save` 也不是原子写。注释说 "card-set hash"，实际是 id-set hash。

---

## 6. 经验更新机制

**【代码】**从 trace 到 patch 的完整路径：

1. `evaluate` 返回 `Results`：matrix、fingerprint、failing、trajectories、`failure_details`、`trial_evidence`、`fingerprint_by_task`（`downstream.py:186-204`）。
2. `write_evidence` 用 `Masker` 掩掉真实 id，把每个失败 task 的 k 条 trial 写成 markdown（含 PASS COUNTERFACTUAL 与 FAIL），并附机制 fingerprint（`driver.py:6617-6687`）。
3. Phase D 的 meta-agent 读 evidence + policy + capability disclosure + 当前包，写 `plan.json`（`driver.py:7502-7587`）。
4. `_prepare_candidate(rnd, working_base)` 复制出一个 disposable candidate（`3287-3300`）；Phase P 只允许编辑候选目录。
5. 验证链：`lint.py`（load/reachability/cfg 合法性，`95-453`）、`sanitize.scan_package`（真实 gold id 红线，`157-165`）、`activation_probe.probe_package`（离线跑真实 selector，确认每张卡至少有一个 carrier 会投递，`361-433`）、`_plan_conformance_errors`（计划与 diff 一致，`8048-8230`）。
6. 接受/拒绝：`gate()` 返回 verdict；`commit()` 才把候选变成 `versions/M{rnd}`（`9677-9750`）；拒绝则丢弃（或按 `--provisional-accept` 临时作为下一轮 working base，`10412-10549`）。
7. 记录：`ledger` 原子事件文件（按 event_id sha256 命名，`driver.py:4539-4601`）、`changelog`、`round_r_lesson.txt`、`round_review.md`、`state.json`。

**状态语义**：

- 有 `superseded` 的等价物，但只在 W：`apply_model_update` 把旧的 `NOT_YET` 改成 `OBSOLETE`，保留审计（`wm/ledger.py:141-158`）。
- **E 没有** `superseded/deprecated` 字段；淘汰一张卡只能 `remove_card`（`plan_schema.py:84-89`）。TTA prompt 里说 "you may supersede it"（`tta/driver.py:336`），实际实现只是再写一张新 id 的卡，旧卡仍在 archive 里。
- 包的版本是目录级不可变（`versions/M*`），不是卡片级版本。

**一个 claim/code 偏差**：`plan_schema.validate_plan` 支持 `blocked_keys`（"do-not-repeat ledger"），`gates.Ledger` 也定义了 `accepted/rejected/untreatable`（`gates.py:99-120`），但 driver 的 `blocked_additions` 永远是 `[]`（`driver.py:10396-10403`），`gates.Ledger` 全仓库无实例化。因此"被拒绝且造成回归的组合不会重试"在 autonomous run 里并未实现；`state["blocked"]` 恒为空。

---

## 7. 失败处理方式

### 7.1 分量级归因

**【代码】**component 枚举与可 patch 集合在 `plan_schema.py:25-32`，动作白名单在 `84-89`：

- `E`: add/edit/remove_card
- `W`: set_wm/set_board/set_grounding
- `RHO`: set/edit/remove_delivery
- `C`: set/remove_checker/set_gate/set_feasibility

`MODEL/HARNESS/BENCHMARK` 只能 `fixes: []`；`CEILING` 是 legacy 非 patchable。`FailureOwner` 在 `domain_adapter.py:51-64` 有同样的分类，`AttemptStatus` 把 `MODEL_ERROR/HARNESS_ERROR/BENCHMARK_ERROR` 映射到外部 owner（`83-98`）。`action_opportunity` 与 `patchability` 是两个独立变量，必须 `O=present ∧ P=memory` 才允许 patch（`plan_schema.py:118-135`）。

这使得"失败处理"首先是**拒答机制**：证据不足时 meta-agent 必须弃权，而不是硬写一张卡。

### 7.2 运行期错误处理

`errors.py` 只有三个异常：`RecurisError`、`WritePermissionError`、`ProfileError`（`errors.py:6-21`）。其余模块各自定义：`PathError`（`paths.py:22`）、`ChampionIntegrityError`（`integrity.py:44`）、`EvaluationProtocolError`（`settle.py:99`）、`EvaluationRunLockError` / `InflightJournalError`（`eval_run_guard.py:17-23`）、`DownstreamRunError` / `DownstreamTimeoutError`（`downstream.py:67-72`）、`BenchmarkProtocolError`（`benchmark_protocol.py:55`）、`MetaAgentSessionFailure` / `MetaTreatmentError` / `ProfileArmIneligible`（`driver.py:638-675`）、`ResultValidationError`（`tools/avgk.py:43`）。

内核把最贵的 bug 写成了结构性防线：模型不能写 DONE（`ledger.py:162-168`）、synthetic id 注册后再 ground 一律拒绝（`runtime.py:271-275`、`grounding.py:153-163`）、unmatched receipt 只计数不伪造 entry（`grounding.py:187-194`）、sanitizer 在所有出站路径剥离伪造 board（`runtime.py:261-264`）。

**但 TB2.1 的 completion 明显更弱**：`TerminalSkillBridge.check_completion` 在 checker 不 bounce 时，把最后一个非 error receipt 当作 evidence，**把所有 pending entry 一次性 mark_done**（`runtime_bridge.py:309-319`）；没有 receipt 时只 `count("completion_allowed_without_evidence")` 仍然放行（`320-322`）。这不是 tau2 的 receipt-binding 语义，不能当作同等强度的 grounding。

---

## 8. Benchmark 设计

| 线 | 入口 | 对照 | 预算/配对 | 冻结方式 |
|---|---|---|---|---|
| τ² retail/airline | `recuris tau2` / `RecurisDownstream` | `llm_agent` vs `recuris_agent` | 同 task 集、同 k、同 worker/simulator model+llm_args、同 treatment triple；每轮 baseline 与 candidate 同时重跑 | split 文件 sha256 入 `split_manifest.json`；driver 把 split sha256 写进 config record（`driver.py:2298-2306`）；train/dev 不重叠校验（`1889-1899`）；`test: []`，运行时不读 test |
| SkillFlow | `recuris skillflow render-configs` + `harbor run` + `score` | 同一 `render_one` 函数，`arm` 只决定有无 template | 同 model/base_url，同 family 任务；paired 按 `family/task`，task-clustered bootstrap 10k（`score.py:79-102`） | 任务集由外部 HF 仓库固定；configs 不提交，每次生成 |
| TB2.1 | `recuris tta run` | `bare` / `m0` / `tta`，同 `--rounds` | 同 taskset、同 seed package、同 image snapshot；`m0` 隔离"有机器的包"，`tta` 隔离"包会更新" | `tta_taskset_v3.json` 预注册 strata A/B/U/P；`split_manifest.json` 把 image snapshot 写成硬前提 |
| 最终 settlement | `recuris metaagent settle` | frozen base vs candidate | 固定 task ids、固定 trial seeds、每 arm 20 cells | 协议字段与常量逐项比对（`settle.py:1376-1403`） |

τ² 细节：retail split 16 train/12 dev/0 test，k=4；airline 11/10/0，k=6（`splits/tau2/split_manifest.json:5-27`）。driver 的 `--k` 必须与 split `fixed_budget.k` 一致（`driver.py:1945-1965`）；`_validate_split_lineage` 还检查 `source_split` 的 sha256 与 `excluded_prior_*` 不重叠（`1901-1944`）。

"bare vs skill 同预算"的真实边界：

- tau2 同的是 **task×trial 预算与解码参数**，不是 LLM 调用数/token：skill arm 每轮多一次 ledger update，bounce 时多一次 redraft（`runtime.py:228-252`、`396-406`）。
- bare 使用 stock `llm_agent`，没有 `wm_pending_lines`/`wm_board_text` 方法；tau2 patch 用 `getattr(self.agent, ...)` 调这些 hook（`third_party/tau2/recuris.patch:664`、`:865`）。所以 `TAU2_GATE_TERM*`/`TAU2_STATUS_BOARD` 对 bare arm 是 no-op，对 skill arm 才生效。**两只 arm 的差异不只是 memory 内容，还包括 harness hook 是否存在**。
- 更严重的是：campaign 的 subprocess 环境过滤掉所有 `TAU2_*`（`downstream.py:622`、`679-695`），之后只补 `TAU2_DATA_DIR` 和可选 `TAU2_RETRIEVAL_CONFIG`，**没有补 `TAU2_GATE_TERM/TAU2_GATE_TERM_WM/TAU2_STATUS_BOARD`**。因此 `metaagent run` 的每次 evaluation 里终端门和状态板都是关闭的（【推断】由 patch 的 `os.getenv` 读取点决定，`recuris.patch:645-675`、`:865`），尽管包的 manifest 写着 `board.status_board: true`、`gate.lines: authorized_only`。演化循环测到的是 agent 侧 checkers/ledger，而不是 README 里描述的那套 harness 机制。

SkillFlow 的公平性设计值得抄：`render_one` 用同一个函数生成 bare/skill，两者只差 `prompt_template_path`（`render_configs.py:51-110`）；`--routing default` 不用任何看过分数的 per-family 选择，`frozen_insample` 明确标注 in-sample 并在 stderr 警告（`templates.py:1-20`、`render_configs.py:170-176`）。这与 README 的 +16.8 需要打折阅读有关。

TB2.1 的诚实点：同一 baseline 在两台 image snapshot 上 34.5% vs 40.0%，所以 strata 只对记录的快照有效（`third_party/tb21/README.md`、`splits/tb21/split_manifest.json:5-11`）；harbor 默认 `--rmi all`，TTA 配置显式 `delete: false`（`tta/driver.py:200`）；连续 3 次没有 `result.json` 触发 breaker，剩余任务记 UNATTEMPTED 而不是 0（`tta/driver.py:125-153`、`435-453`）。

**无法验证的部分**：README 的 results 表（35/37 对、+23.3 等）在仓库里没有对应的 runs/artifacts；`git ls-files` 只有 210 个文件，没有 results 目录。所有数字只能视为作者声明，本审计不做复现。

---

## 9. 可复用组件（接口签名）

按"低耦合、可直接搬"排序：

1. **`TurnRuntime` + `HarnessPort`**（`runtime.py:47-62`、`121-266`）。Port 需要实现 `incoming_kind/user_text/extract_receipts/genuine_user_texts/read_tool_results/ledger_llm/llm_draft/redraft_with_correction/bounce_and_redraft/commit/draft_content/draft_tool_calls/set_draft_content`。这是把任意 agent 循环接到 kernel 的唯一缝，EvoMemory 的 harness 可以直接实现这一组。
2. **六个 SPI Protocol**（`spi.py:47-210`）：
   - `EntryKind.parse_proposal(raw, ctx) -> dict|None`、`maintain_template() -> str`、`render_entry(entry, index, schema) -> str`、`render_texts() -> dict`、`pending_line(entry, schema) -> str`
   - `WMManager.should_update(ctx) -> bool`、`build_instruction(wm_view, profile) -> str`、`parse_reply(raw) -> list`
   - `Deliverer.select(ctx, wm_view, em, draft=None) -> Sequence[DeliveryAction]`
   - `GroundingMatcher.match(evidence, pending) -> str|None`、`relevant(evidence) -> bool`
   - `Checker.inspect(ctx, wm_view, draft_text, draft_has_calls) -> Bounce|None`
   - `FeasibilityOracle.rule(pending, evidence) -> Sequence[BlockRuling]`
3. **`Ledger` 的写权限模型**（`wm/ledger.py:112-201`）：`apply_model_update(requests, write_tools, ctx)`、`mark_done(entry_id, evidence_id, writer)`、`mark_blocked(entry_id, reason, writer)`、`mark_authorized(entry_id, evidence, writer)`。这是"模型只能提议、harness/oracle 才能写状态"的最小实现，和具体 benchmark 无关。
4. **`GroundingKernel.ground(receipts, consumed) -> GroundingReport`**（`grounding.py:165-195`）。admission 与 matching 分离；matcher 只能决定"哪条 entry"，不能决定"是否准入"。`ToolReceipt(call_id, tool, args, error, synthetic, content)` 是通用 evidence 形状，可扩展成多模态 evidence。
5. **`Fingerprint`**（`fingerprint.py:21-54`）：`count(name)`、`event(kind, **payload)`、`dump(path)`、`aggregate(jsonl)`。机制先于分数，这个设计可以原样用于 EvoMemory。
6. **`held_out_paired_gate(base, cand, alpha=.05, reg_cap=0, eps=1e-9, n_boot=3000, seed=0, material=0.0)`**（`gates.py:31-75`）。输入是 `dict[item -> [per-seed score]]`，返回 `Verdict(accept, net, ci, n_up, n_down, reason)`；对 item 重采样、不依赖任何模型。可直接作为 EvoMemory 的 admission gate。
7. **`integrity.verify()` / `assert_mutation_paths_safe(paths)`**（`integrity.py:221-270`）：内容寻址的 champion lock + 写路径越界检查。EvoMemory 若允许多模态记忆文件，可以用同样机制保护"已结算记忆"。
8. **`skillmemory.load_skill_memory(path)`**（`skillmemory.py:167-223`）：load-time 校验未知 name/缺 SPI 方法/非法 local import（`_load_local` 用锁 + 路径必须在包内，`51-86`）。适合把"记忆包"做成纯数据。
9. **`activation_probe.probe_package(package, domain, benchmark) -> dict`**（`activation_probe.py:361-433`）：离线实例化真实 runtime，对每张卡跑真实 selector，报告 `activated`。这个"先证明记忆会触发，再测效果"的顺序非常值得复用。
10. **`eval_run_guard.RunFileLock`**（`eval_run_guard.py:25-87`）：跨进程字节锁 + inflight journal，Windows/POSIX 都实现。

---

## 10. 改造难点

**【代码】强依赖：**

- tau2：`adapters/tau2/agent.py:21-30` 直接 import `tau2.agent.llm_agent.LLMAgent`、`tau2.data_model.message`、`tau2.environment.tool`、`tau2.utils.llm_utils`；`run.py:180-213` import `tau2.registry`/`TextRunConfig`/`run_domain`。`downstream.py` 的 `_benchmark_git_commit` 要求 `TAU2_ROOT` 是真实 git clone（`121-135`），且 `RecurisDownstream` 的默认模型/args/domain 全是 tau2 语义。
- Driver 只支持 `SUPPORTED_BENCHMARKS={"tau2"}`（`driver.py:147`），`SUPPORTED_TAU2_DOMAINS={"retail","airline","banking_knowledge"}`（`142`）；`BenchmarkProfile` 写死 `package_prefix="tau2"`、`skill_agent="recuris_agent"`、`bare_agent="llm_agent"`（`531-558`）。虽然留了 `DomainAdapter` Protocol（`domain_adapter.py:701`），但 driver 的构造/评估路径没有接它。
- 轨迹/评分格式：`Results.matrix/failing/seeds/trial_evidence` 和 `reward_info`（`action_checks/db_check/nl_assertions/communicate_checks`）是 tau2 形状（`downstream.py:206-305`）。
- SkillFlow：harbor==0.20.0 + Docker Compose V2；`NoInstallQwenCode` 的 `run()` 覆盖了 template hook，需要打 patch（`third_party/skillflow/README.md`）。TB2.1：harbor + Terminus-2 内部 hook（`_handle_llm_interaction`/`_execute_commands` 签名随版本变化，`harbor_terminus.py:52-56`）。
- 演化循环：Claude Code CLI（>=2.1.226，`launchers/README.md`）、Anthropic→OpenAI proxy（`fastapi/uvicorn/httpx`）、`RECURIS_META_MODEL` 等环境变量；meta-agent 的每次 edit 都要求 `Read/Edit` 工具面，`cc_session` 还审计 read/write scope 与 upstream model（`driver.py:5584-5880`）。
- 路径：`repo_root()` 必须找到同时含 `skill_memories/`、`splits/`、`third_party/` 的 checkout（`paths.py:31-66`）；wheel 安装不可用。

**接到自建小型 benchmark 的最小改动路径：**

1. 只复用 kernel（`TurnRuntime` + `WMSchema` + `EMStore` + `Ledger` + `GroundingKernel` + `Fingerprint`），不碰 `metaagent/driver.py`；写一个 `HarnessPort` 适配器（约 12 个方法）。
2. 定义新 `EntryKind`（或复用 `generic_item`/`knowledge_need`），manifest 用 `_base` 起步（`skill_memories/_base/manifest.yaml`）。
3. 若要用演化循环：实现一个 `RecurisDownstream` 等价 shim，返回 `Results(matrix, failing, seeds, trajectories, failure_details, trial_evidence, fingerprint_by_task)`，并提供 `dense_rewards`；再给 `BenchmarkProfile`/`validate_plan` 提供 tool capability 与 event dispatch 表。`plan_schema` 的动作白名单与 `lint.py` 的 carrier 规则是按 tau2 工具语义写的，需要按新 benchmark 重写或放宽；`EMEntry/ToolReceipt/Sanitizer/Fingerprint` 都是 `str` 假设，多模态需要改 schema。
4. 去掉 tau2 相关导入后，kernel 的 `pyyaml+loguru+numpy` 足够；不需要 Docker、vLLM、FastAPI。整个替换的关键不是模型，而是"evidence 形状 + carrier 可达性 + 评分矩阵"三件事。

---

## 11. 许可证、third_party、依赖

### 许可证

- 本仓库 **Apache-2.0**：`LICENSE:1-3`，版权行 `LICENSE:189` "Copyright 2026 The Recuris Authors"；`THIRD_PARTY_NOTICES.md:3` 明确 "Nothing third-party is redistributed"，`third_party/` 只有 pin、patch、setup 脚本，组件本体在 `external/`（git-ignored）。
- `tau2`：MIT，© 2025 Sierra Research，pin `8ebb749`，数据 tag `v1.0.1`，patch 改 11 个文件（orchestrator hooks、judge/env 常量、OpenAI-compatible embedder）；MIT 允许再分发，但仓库选择 fetch+patch。`harbor`：Apache-2.0 0.20.0，patch 只改 Docker exec/upload 两个文件。`tb21`：Apache-2.0，pin `5c8eadf`，91 tasks，未修改；镜像不随仓库分发。
- `skillflow`：**无许可证**，all rights reserved；仓库不包含任何 SkillFlow 行，只提供 patch 和 setup。**这对 EvoMemory 是硬约束：不能复制它的代码/配置/模板文本。**
- 演化循环的 coding agent：专有软件，不随仓库分发，只给 launcher contract。

结论：**可以借鉴 Recuris 自己的 Apache-2.0 代码（保留 notice）；不能把 SkillFlow 的任何内容搬进 EvoMemory；tau2/harbor/tb21 的代码可以按 MIT/Apache 使用，但要保留各自 notice，且镜像/任务数据另有版权。**

### 依赖重量（`pyproject.toml:24-67`）

- 基础依赖只有 `pyyaml>=6.0`、`loguru>=0.7`；`tau2` extra 加 `pydantic/numpy`；`metaagent` 再加 `openai/httpx/fastapi/uvicorn`；`skillflow`/`tb21` 加 `jinja2` + `harbor==0.20.0`（钉死版本，因为 patch 针对它）；`dev` 加 `ruff`；`all` 聚合（`pyproject.toml:24-67`）。
- 运行时外部依赖：Python >=3.12、git、Docker + Compose V2（SkillFlow/TB2.1）、vLLM（本地开源模型 + tool-call parser）、Claude Code CLI（演化循环）、OpenAI-compatible endpoint（judge/simulator/meta）。没有 SQLite、没有向量数据库；embedding 只用于可选 deliverer，且 lazy import vendor SDK（`embed_retrieval.py:27-33`）。

---

## 12. 可疑点（stub / 硬编码 / 声明未落地）

1. **`field_writers` 是死字段**：`schema.py:87-96` 声明，全仓库无读取；实际权限硬编码在 `ledger.py:162-201`。
2. **`leakage_check`、`fingerprint_verify` 是死函数**：`gates.py:78-96` 只有定义；driver/settle 只 import `held_out_paired_gate`（`driver.py:81`、`settle.py:43`）。泄漏实际由 `sanitize.scan_package` 做（`lint.py:108-122`、`driver.py:10045`）；机制指纹只作为 `mechanism_delta` 报告（`driver.py:9587-9592`），不参与 `relaxed/progressive` 的 accept 条件。`gates.py:1-15` 自称"four gates, all of them code"，实际只有 1 个被主循环使用。
3. **do-not-repeat ledger 未接线**：`blocked_additions` 恒为 `[]`（`driver.py:10400`），`gates.Ledger` 无实例化；`state["blocked"]` 永远空。
4. **`configs/` 不被任何 runner 读取**：`recuris.config` 只被 CI 调用（`.github/workflows/ci.yml:71-78`），CLI 的 `tau2`/`metaagent` 全走 flags；README 的 "use the shipped configs"（`docs/operations.md:24-26`）在代码里没有对应加载路径。
5. **SkillFlow 包不是可加载的 Recuris 包**：`skill_memories/skillflow/manifest.yaml:29-37` 使用 `standing_inject` 的 `em_type: machine`、`need_driven_retrieval` 的 `granularity/audit`、checker `pre_submit_contract_assert`；`builtin.__init__.py:28-51` 没有这个 checker，`deliverers.py:107-119` 没有这些 cfg key，`load_skill_memory` 会 `ProfileError`/`TypeError`。它自己也在 `:1-20` 声明 "ADAPTER IS EXTERNAL"、"validation-pending"；`em/` 目录不存在。也就是说，skillflow 线不是通过 kernel 跑的。
6. **缺失的引用文件**：skillflow manifest 指向 `docs/DESIGN_WM_env_interaction_2026-07-07_CN.md`、`docs/HANDOFF_TB_SkillFlow_2026-07-07_CN.md`、`skiller-framework/tools/gen_skillflow_templates.py`，三者都不在仓库（`skill_memories/skillflow/manifest.yaml:15-17`）。
7. **airline 的 board_marker 多一个 `---`**：manifest 第 13 行 `"--- Progress (system-verified) ---"`，`BoardRenderer.render` 又拼 `f"{self.marker} ---"`（`board.py:49`），实际表头会出现两个分隔符；不影响功能，但说明包与渲染器之间的字节契约没有测试。
8. **campaign 里 TAU2_* 被清空**：见第 8 节；`treatment_triple` 只在 standalone arm 打印/记录（`treatment.py:163-188`、`run.py:166-178`），campaign 的 `RecurisDownstream` 不设置它们。包里的 `board`/`gate.lines` 在演化评估中不生效。
9. **`runtime.py` 的 `_ground` 允许"同 tool 无 binding 也匹配"**：`ReceiptBindingMatcher._score` 在 `bind` 和 `ebind` 都为空时仍给 `tool_match_bonus=1`，而 `min_score=1`（`grounding.py:96-122`、`schema.py:49-56`）。多个同 tool pending entry 且 args 缺 binding key 时，会按列表顺序结算第一个。文档说的"conflicting binding key excludes"只覆盖了有值冲突的情况。
10. **TTA 的 leak flag 只记录、不拒绝**：`tta/driver.py:479-484` 把命中 `\b\d{4,}\b` 的 card 写进 `rec["leak_flags"]`，然后**照常写卡**；而 campaign 的 leak scan 是红线拒绝（`lint.py:108-122`）。两者口径不同。
11. **没有测试**：仓库没有 `tests/`；CI 只做 ruff、integrity、import、config validate（`.github/workflows/ci.yml:1-90`）。多处注释声称 "byte-parity tests keep guarding them"（`builtin/entrykinds.py:1-5`、`wm/render.py:78`），但测试没有随仓库发布。
12. **硬编码**：doubao 模型 id（`driver.py:106-108`、`downstream.py:53`、`tta/driver.py:73-82`）、airline oracle 的固定时钟 `2024-05-15T15:00:00`（`skill_memories/tau2_airline/plugin.py:20`）、settle 的 task 38/50/33/36/86 与 seeds（`settle.py:51-96`）、scorecard 的 retail/airline/banking 写工具集合（`tools/scorecard.py:22-44`）、`SUPPORTED_TAU2_DOMAINS`（`driver.py:142`）。这些是"研究产物"而非通用库，符合定位，但移植时要全部替换。
13. **`pyproject.toml` 的小不一致**：`requires-python >=3.12`（`pyproject.toml:10`），但 `[tool.ruff] target-version="py311"`（`:95`）；不影响运行，说明发布时没有统一。README 的结果表也没有 run artifact 可复现；`docs/operations.md:49-51` 自认 judge 不是 leaderboard 的 judge，绝对分数不可跨论文比较。

---

## 13. 并发与状态文件竞争

**【代码】**没有 `multiprocessing`；并发点是线程 + 子进程：

- `driver` 用 `ThreadPoolExecutor` 并行 diagnosis workers（`driver.py:6867-7060`），`tta/driver.py:586-602` 用线程池跑 harbor，`Fingerprint` 用进程内 `_IO_LOCK` 保证 jsonl append 不交错（`fingerprint.py:18`、`51`）；`skillmemory._load_local` 用模块级 `_LOCAL_IMPORT_LOCK`，并改为"先 exec 完整、再注册 sys.modules"，修掉了 cc>1 时半初始化模块的竞态（`skillmemory.py:45-86`）。
- `Driver` 用 `.driver.lock` + `logs/.campaign.lock` 做跨进程非阻塞字节锁，禁止同一 run 或同一 framework tree 上并发 campaign（`driver.py:1769-1827`）；可用 `RECURIS_ALLOW_CONCURRENT_CAMPAIGNS=1` 绕过。
- `settle` 用 `RunFileLock`（msvcrt/fcntl）+ inflight journal，要求"先证明已提交、再清 inflight"（`eval_run_guard.py:25-189`）。
- `_write_json_atomic`/`_write_text_atomic` 都是同目录 tmp + `replace`（`driver.py:2475-2497`）；`_finalize_round` 用 `round_r_finalize.json` 幂等日志（`4631-4669`）；`commit` 先 copytree 到 `.M{rnd}.tmp`、校验 hash、再 rename（`9677-9750`）。

**仍然存在的竞争/一致性风险：**

- `state.json` 的 `save_state()` 没有锁；driver 主线程写、worker 线程不写，当前安全；但未来若把 round 并行化会立刻坏。
- `Fingerprint._IO_LOCK` 是**进程内**锁；两个进程写同一个 `RECURIS_FINGERPRINT` path 时 jsonl 仍可能交错，`scorecard.py:220-224` 明确警告 "fp lines != results sims"。
- `embed_cache` 的 `.npy` 是 `np.save` 直接写，不是 tmp+rename；两个 evaluation 共享同一 cache 时可能读到半写文件。下游用 per-namespace + package digest 隔离（`downstream.py:689-692`），standalone 默认 `.embed_cache` 没有隔离。
- TTA 的 `_infra` 是全局 breaker，一个任务的连续基础设施失败会停止剩余任务（`tta/driver.py:125-153`），summary 用 `n_unattempted` 显式暴露；`shutil.rmtree(package)` 只在 `resume_from==1` 执行（`:428-432`），不会删除已完成 round 的 archive，但如果两个 `run_task` 线程误指向同一 task dir 没有锁（目录名来自 task name，正常不会重复）。

---

## 14. 对 EvoMemory 的启示

### 值得吸收（3–6 条）

1. **"机器 / 记忆"目录级分离 + 数据化 manifest**。`src/recuris/` 是不变的 turn loop，`skill_memories/` 是纯 YAML+Markdown；新增知识 = 新增一个 `.md`，不改代码（`skill-memory-format.md:6-9`、`skillmemory.py:167-223`）。EvoMemory 可以把 `E` 扩展成 `(text, image, audio, video)` 的 entry 目录，`W/ρ/C` 仍用声明式配置，这样多模态记忆不会把 harness 拖进模型代码。
2. **写权限三通道 + evidence admission 与 matching 分离**。模型只能 `apply_model_update`，`DONE/BLOCKED/AUTHORIZED` 只能由 HARNESS/ORACLE 写；synthetic id 注册后再 ground 一律拒绝；matcher 只能选 entry，不能决定准入（`wm/ledger.py:1-14`、`grounding.py:137-195`、`spi.py:153-166`）。这套结构可以直接用于多模态证据：把图片/音频的 `receipt_id + hash + modality` 作为 `ToolReceipt`，模型永远不能自己宣布"证据已看过"。
3. **item-level paired bootstrap + `reg_cap` + power gate**。不重采样 trial，避免伪窄 CI；`reg_cap` 把"回归个数"变成硬约束；power gate 在开跑前用 MDE 拒绝欠功效设计（`gates.py:31-75`、`driver.py:9975-10029`）。EvoMemory 的小型 benchmark 更应该先算 MDE，否则 5–10 个任务上的单轮提升根本不可检测。
4. **机制指纹先于分数 + 激活探针**。`Fingerprint` 记录每个 deliverer/checker 的实际触发，scorecard 拒绝在机制计数之前打印分数（`fingerprint.py:1-8`、`scorecard.py:197-250`）；`probe_package` 在离线状态用真实 selector 证明"这张卡会被投递"（`activation_probe.py:361-433`）。多模态检索尤其需要这个：检索器可能"看起来有结果"但根本没进模型上下文。
5. **内容寻址的不可变记忆版本 + 完整 provenance**。`versions/M{rnd}`、`package_tree_sha256`、`champion lock`、evaluation artifact 里的 model/commit/input/seeds/args 全记录（`driver.py:9677-9750`、`integrity.py:221-254`、`downstream.py:623-670`）。EvoMemory 的记忆会跨模态、跨模型演化，必须能回答"这条记忆是在哪个模型、哪个数据快照、哪次 run 上被接受的"。
6. **O/P 准入与"弃权是合法结果"**。`action_opportunity=present ∧ patchability=memory` 才允许 patch，否则 `fixes: []`（`plan_schema.py:118-135`、`diagnosis.md:17-24`）。这个约束能显著降低"为了演化而演化"的假 patch，尤其适合多模态失败归因（很多失败其实是渲染/工具/评测器问题）。

### 不该照搬（2–3 条）

1. **11k 行的 `metaagent/driver.py` 及其外部依赖链**。它把 tau2、Claude Code、Anthropic proxy、固定 split、固定 task id、doubao 模型、Docker/harbor 全焊在一起；真正可复用的 gate/ledger/fingerprint 只占很小一部分。EvoMemory 应重写一个 300–500 行的 round loop，复用 `gates.py`/`integrity.py`/`runtime.py`，不要移植 `Driver`。
2. **文本专用的卡片/清洗/truth 机制**。`EMEntry.body:str`、`Sanitizer` 的字符串 marker、`TruthProtocol` 的英文/中文正则、`leakage_check` 的字符串包含，都不适合图像/音频/视频；而且这些机制在代码里大量依赖"卡是自然语言"这一前提。EvoMemory 需要 modality-aware 的 sanitizer 与 evidence schema，不能把 Recuris 的正则当通用层。
3. **"文档声称、代码未接线"的部分 + TTA 的弱口径**：`field_writers`、`leakage_check`/`fingerprint_verify`、`blocked` do-not-repeat ledger、`configs/` 加载、campaign 里的 `TAU2_GATE_*` 都未真正生效；`runtime_bridge.py:309-319` 用一条 receipt 结算所有 pending，`tta/driver.py:479-484` 对泄漏只记录不拒绝。这些是发布时的缺口与妥协，不是设计思想；EvoMemory 如果照抄文档而不写测试，会重复"看起来有机制、实际没运行"的问题，应以"每个机制有一个能在 CI 里跑的断言"为准入条件。
