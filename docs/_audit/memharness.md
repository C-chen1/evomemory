# MemHarness 源码审计

审计对象：`references/memharness`（verl fork，重点 `agent_system/`）。`P:` = 相对 `references/memharness/` 的路径。
只读审计，未运行任何代码。**【代码】**＝实际读到的实现；**【README】**＝文档声称；**【推测】**＝无直接证据。

## 0. 一句话结论

代码层面它不是"五阶段 agent"，而是**"检索 + 一次额外的 LLM 改写 pass + 把结果塞回 observation"**；
critique 与 reconstruction 合并为同一次生成，判定信号只有字面量 `<EMPTY>`。
"reconstruct 而非 replay"的**机制**值得复用；"靠 GRPO 涌现改写能力"的**主张**在发布脚本里**没有对应损失项**（§10）。

## 1. 核心问题与主张：replay vs reconstruct

**【README】**（`README.md:17-21,40-43,59-62`）：replay 把抽象经验原样注入具体状态，造成负迁移；
MemHarness 在 retrieval 与 action 之间插入 critique+reconstruction，由**统一策略**改写为 state-aligned guidance，
能力"通过 GRPO 端到端涌现"。报告 85.2 / 75.6（ALFWorld / WebShop），比 pure GRPO +8.8 / +9.5，OOD 85.9，
关掉记忆仍有 83.0。

**【代码】** 实际是四段；`agent_system/` 下搜索 `critique` / `reconstruct` **0 处**（仅 README 出现）：

| README 阶段 | 代码对应 | 证据 |
|---|---|---|
| observation | `build_text_obs` 组装 prompt | `P:agent_system/environments/env_manager.py:275-316` |
| retrieval | `retrieve()` + `<memory>…</memory>` 注入 | `P:agent_system/memory/memory_manager.py:404-452`；`P:agent_system/environments/base.py:216-224` |
| critique | 同一次 adaptor 生成输出 `<EMPTY>` | `P:agent_system/memory/mem_adaptor_rollout.py:68-101` |
| reconstruction | 同一次生成的非空文本，回填 memory 块 | 同上 `:407-454,460-537` |
| action | 策略正常 rollout | `P:agent_system/multi_turn_rollout/rollout_loop.py:444-464` |

**replay / reconstruct 的唯一机器可辨差异**：注入前块内是 `Situation:` + `Principle:`
（`memory_manager.py:535-569`）；adaptor 之后被替换成 "Adapted memory principle 1: …"
（`mem_adaptor_rollout.py:431-454,535-537`）。同一个容器，不同内容。

## 2. 关键算法

### 2.1 检索
- **嵌入**：OpenAI 兼容 `/v1/embeddings`，默认 `bge_m3`、1024 维（`P:verl/trainer/config/ppo_trainer.yaml:473-477`；
  `P:agent_system/memory/milvus_store.py:52-94`）。无本地 fallback：`embedding_api_url` 为空时构造期 `ValueError`（`:63-64`）。
- **检索键**：默认 `retrieve_key=memory_text`（`ppo_trainer.yaml:478`），**用当前 observation/模型 query 匹配抽象原则的向量**，
  不是 state→state；代码自己警告这种不对称（`memory_manager.py:114-123`）。
- **top-k / 阈值**：`top_k=3`、`min_retrieval_score=0.0`（`ppo_trainer.yaml:480-481`）——**没有有效阈值**（代码默认 0.1 被 yaml 覆盖）。
- **相似度**：Milvus `COSINE`+`FLAT`（`milvus_store.py:820-822`），`score = 1 - distance`（`:417`），即余弦；FLAT 为暴力精确检索。
- **过滤**：`only_successful` 默认 **False**（`ppo_trainer.yaml:482`）；True 时加 `success == true`（`milvus_store.py:409-410`）；
  检索侧不加 task 过滤（task 隔离靠 collection 名，`:45-49`）。
- **去重**：插入时探邻居、检索时先 prefetch 再贪心去重（`:199-223,439-445`；算法 `P:agent_system/memory/memory_text_dedupe.py:26-37`）。
  yaml 默认 0.85（`ppo_trainer.yaml:486-489`），但**训练脚本显式置 null 关掉**（`P:run_scripts/train_alfworld.sh:135-136`）。
  README 的 "semantic deduplication" 在发布配方里未启用。

### 2.2 critique 判定
不是打分，是**字符串协议 + 一次 LM 生成**（`mem_adaptor_rollout.py:68-101`）：
空输出 → reject；严格等于 `<EMPTY>`/`<empty>`（`ppo_trainer.yaml:631-633`）→ reject；
含 `<action>`/`<memory_retrieve>`/`<memory>`/`assistant` → reject（防共享权重时续写 agent 协议，`:90-95`）；否则 accept。
无数值阈值参与决策；`log_adaptor_state_similarity` 只把 S_old/S_curr 余弦**记日志**（`mem_adaptor_rollout.py:706-720`、
`P:agent_system/memory/state_embedding.py:49-93`）。`schedule=on_memory_only` 时仅当本步有注入/命中才跑（`:202-207,634-645`）；
`retrieval_top_k=1`，默认只改写 top-1（`ppo_trainer.yaml:557`）。

### 2.3 reconstruction 输出格式
Prompt（`ppo_trainer.yaml:610-630`）：system 要求"简洁可复用 guidance 或 `<EMPTY>`；无 CoT、无第一人称、无分步计划"；user 给
`{task}{s_curr}{s_old}{p_old}`。输出 = header + `"Adapted memory principle {index}:\n"` + 正文，多命中 `\n\n` 连接
（`mem_adaptor_rollout.py:431-454`；`ppo_trainer.yaml:635-638`）。

### 2.4 fallback
- 检索为空 → 注入 `empty_retrieval_message`（"No validated memory principle applies; rely on observation and reasoning."），
  记 `retrieved=[]` 的 `MemoryEvent`（`memory_manager.py:432-440`；`ppo_trainer.yaml:405`）。
- adaptor 全部 reject → **删除整个 memory 块**再贴回同文案（`mem_adaptor_rollout.py:512-520`）。
- 无命中且无注入 → adaptor 不运行。fallback 不是"策略重新推理"，只是把记忆拿掉、原 prompt 生效。
- **负迁移抑制实际有四道闸**：空检索→中性提示；accept/reject 闸门；插入/检索去重（默认关）；utility 剪枝（默认开）。
  真正起作用的是 1/2/4；2 是唯一 state-conditioned 判断，且由 LM 完成。

## 3. 数据模型（types.py）

```python
MemoryRecord(memory_id, task_name, item_id, source_episode_id, source_step,
             state_text, action_text, memory_text, reward, success,
             created_step, created_at, retrieval_count, last_used_step,
             metadata: dict, value, value_source, value_update_step)   # types.py:12-38
RetrievedMemory(memory_id, score, state_text, action_text, memory_text, reward,
                metadata, value, value_source)                          # :41-54
MemoryEvent(event_type, query_text, state_text, injected_text, retrieved[]) # :57-66
```

- `memory_id = uuid4`（`P:agent_system/memory/experience_summarizer.py:1124`），非内容哈希；`state_text`（历史情境）与
  `memory_text`（可复用原则）分离，是 reconstruct 成立的前提。
- `metadata` 写入 `dataset_item_id / trajectory_index / memory_index_within_trajectory / extracted_by /
  experience_writeback_kind / extraction_schema`（`:1099-1107`）+ utility 计数 + `llm_suggested_value`。
- **无模态字段**：没有 image/video/URI/证据指针。初值 `value=0.5`、`value_source="utility_score_prior"`（`:1148-1150`）。
  长度约束 1200/400/1200（state/action/memory，`ppo_trainer.yaml:520-522`），`normalize_text` 折空白后截断（`memory_manager.py:28-37`）。

## 4. 执行生命周期（函数级）

```
rollout_loop.vanilla_multi_turn_loop                    rollout_loop.py:335
 ├ envs.reset → build_text_obs → _pair_finalize_memory  base.py:136-150
 │   ├ fixed   : build_memory_message_and_event(S_curr) memory_manager.py:404
 │   └ agentic : append_retrieval_hint(prompt)          memory_manager.py:383
 ├ preprocess_batch → actor_rollout_wg.generate_sequences
 ├ envs.step_with_memory(...)                           env_manager.py:318 / 734
 │   ├ _extract_memory_queries (<memory_retrieve>)      memory_manager.py:389-398
 │   └ 有 query → _inject_memory_queries_into_observations   base.py:160-188
 │        └ build_memory_message_and_event → store.retrieve → _format_memory_prompt
 ├ maybe_apply_memory_adaptor(...)                      mem_adaptor_rollout.py:556
 │   ├ _retrieved_state_principle_pairs (S_old,P_old)   :104-120
 │   ├ _build_adaptor_prompt → DataProto → generate     :324-404, 725-735
 │   ├ _replacement_from_many_raws (accept/<EMPTY>)     :407-454
 │   └ _patch_one_obs_text → _splice_out_memory_block   :494-537, 460-491
 ├ 记录 memory_query_mask / memory_injected_text / memory_retrieval_counts  rollout_loop.py:487-517
 └ episode 结束
   ├ success_evaluator                                  base.py:226-273
   ├ maybe_write_rollout_memories                       rollout_loop.py:604-617 → memory_manager.py:458
   │   ├ experience_summarizer (self|teacher)           experience_summarizer.py:1154
   │   ├ store.add_records                              experience_summarizer.py:1428
   │   ├ _apply_episode_utility_updates                 memory_manager.py:279-318
   │   └ _maybe_run_experience_utility_maintenance      memory_manager.py:320-381
```

注入在 env step 后、下一轮生成前；adaptor 在注入后、action 前；写回在 episode 后，且 `is_train=False` 不写回
（`rollout_loop.py:602-604`）。`memory_query` 默认算一步（`count_query_as_step=True`，`ppo_trainer.yaml:465`），
安全上限 `max_steps+6` 轮（`rollout_loop.py:396`）。

## 5. 存储策略

- **实现**：`MilvusMemoryStore`（Milvus Lite，文件 `milvus_memory.db`）。`auto_start_server=True`（默认）时主进程起
  本地 FastAPI 子进程，并把配置改成 `backend=http`+`vdb_base_url`（`P:verl/trainer/main_ppo.py:146-160`；
  `P:agent_system/memory/local_service.py:150-162`），worker 走 HTTP 而非各自直连 Lite 文件。
- **collection**：`agent_memories_{sanitized(task_name)}`（`milvus_store.py:45-49`），`auto_id=False`。
- **schema**（`:792-812`）19 字段：主键 `memory_id VARCHAR(128)`；`memory_vector FLOAT_VECTOR(dim)`（字段名由 `retrieve_key`
  推导，`:767-770`）；`state_text/action_text VARCHAR(4096)`；`memory_text/metadata VARCHAR(8192)`；
  `reward FLOAT`、`success BOOL`、`retrieval_count INT64`、`value FLOAT`、`created_at` 等。
- **索引**：`FLAT` + `COSINE`（`:820-822`）。没有 HNSW/IVF/PQ——精确检索，10^3~10^4 可用，再大即线性扫描。
- **更新**：`update_records` = 按 id 查整行 → 改 → delete → re-insert（`:497-611`）；且**每次检索对每条命中同步 update**
  （`memory_manager.py:254-277`）。一次 retrieval ≈ 1 search + k 次 HTTP update + k 次 delete+insert，是最大吞吐瓶颈。
  去重代价：插入时把 `probe_limit` 个邻居的 `memory_text` 重 embed 再比余弦（`:199-223`），每次插入额外 N 次 embedding。

## 6. 经验更新机制

**写回**（`experience_summarizer.py`）：`mode=self` 走 `actor_rollout_wg.generate_sequences`（`:785-836`）；`mode=teacher`
走 OpenAI 兼容 HTTP+线程池+重试（`:659-730`）；发布脚本用 self（`train_alfworld.sh:44`）。schema `full|compact`（`:39-128`），
脚本用 compact。每轨迹最多 1 条、每 episode ≤`max_pairs_per_episode=8`（`ppo_trainer.yaml:416,423`）；长轨迹 head10+tail10+
保护 `<memory_retrieve>`+token 裁剪（`:404-458`）；写回前按 GRPO group 均衡子采样 `grpo_summarize_keep_fraction=0.5`（`:1286-1305`）。
**注意**：`only_store_positive_reward` 默认 False（`ppo_trainer.yaml:415`），**失败轨迹也会入库**，
与 README"successful trajectory"叙事不符（写回 prompt 仍写 "The trajectory is successful"，`:56`）。解析失败重试 3 次后丢弃（`:1377-1384`）。

**utility**（`P:agent_system/memory/experience_utility.py:14-16`）：`score=(c_succ+1)/(c_use+2)`，Beta(1,1) 先验，初始 0.5。
检索时每个返回 id `c_use+=1`（`memory_manager.py:254-277`）；episode 成功时该轨迹**任何一步检索过的所有 id** `c_succ+=1`
（`:279-318` + `experience_utility.py:39-56`）。不是因果归因，只是"出现过且最终成功"的关联计数。分数同步写 `value/value_source`
（`milvus_store.py:558-568`）。

**剪枝**（`memory_manager.py:320-381`）：按 trainer step 周期触发；`prune_every_n_global_steps`（yaml 0=关；脚本 20）、
`prune_score_threshold`（yaml 0.35；脚本 0.3）、`min_uses_before_prune=3`。实现为全表 `query_iterator`，
`c_use>=min_uses && score<threshold` 才删（`milvus_store.py:613-670`）。`clear_every_n_global_steps` 可整库 drop+recreate（默认 0）。

## 7. 失败处理

| 场景 | 代码行为 | 证据 |
|---|---|---|
| 检索 0 命中 | 注入中性 fallback，记 `retrieved=[]` | `memory_manager.py:432-440` |
| 检索抛异常（VDB/embedding 挂） | **不捕获**，穿过 `step_with_memory` → rollout 失败 | `memory_manager.py:425-429`；`env_manager.py` 无 try/except |
| 记忆服务启动失败 | `RuntimeError`，训练初始化阶段退出 | `P:agent_system/memory/local_service.py:227-252`；`main_ppo.py:151` |
| adaptor 输出 `<EMPTY>` | 删块 + 贴 fallback | `mem_adaptor_rollout.py:512-520` |
| adaptor 输出像 agent 协议 / 生成异常 | 前者当 reject；后者 `generate_sequences` 无 try（`:732`） | `:90-95` |
| summarizer 异常 / JSON 解析失败 / add_records 失败 | 前两者 catch 或重试 3 次后丢轨迹；后者 catch，0 条，训练继续 | `:1377-1384,1393-1396,1446-1449` |
| 剪枝/清空失败 | catch，打日志，继续 | `memory_manager.py:334-341,360-366` |
| embedding 未配置 | 构造期 `ValueError` | `milvus_store.py:63-64` |

结论：**读路径硬失败、写路径软失败**。生产化必须先给 retrieve 加超时/降级（退化为空记忆）。

## 8. Benchmark 设计

- **ALFWorld**：`AlfredTWEnv`、`max_steps=50`、`rollout.n=8`、train batch 16、200 步、lr 1e-6、KL 0.01、TP=2、8 GPU
  （`train_alfworld.sh:104-118,218-295`）。val 140、每 5 步测、`val_before_train=True`。
- **In-dist / OOD**：`env.alfworld.eval_dataset` ∈ {`eval_in_distribution`, `eval_out_of_distribution`}
  （`ppo_trainer.yaml:529-532`；`env_manager.py:1013-1018` 透传）。**OOD 直接复用 ALFWorld 自带 unseen-layout split，
  无自建划分**；README 的 85.9% 即此（`README.md:42,227-229`）。
- **WebShop**：`use_small=True`、val 250（`train_webshop.sh:37,97`；`ppo_trainer.yaml:543-545`）。
- **pure GRPO 对照**：脚本层面 = `MEMORY_ENABLED=False`（实验名后缀 `no_memory`，`train_alfworld.sh:79-84`）。
  **但** `FORMAT_REWARD_CLI` 无条件 `require_memory_retrieve=True`（`:188-196`），关记忆后格式奖励仍要求输出
  `<memory_retrieve>`，而检索提示因 `enabled=False` 不再追加（`memory_manager.py:383-387`）。
  **该对照不干净**：要么另有未发布脚本，要么 baseline 被格式奖励污染。复现时必须同步置 `require_memory_retrieve=False`、
  `weight_memory=0`。
- 记忆库训练中持续增长、跨 checkpoint 共享；无固定快照/冻结评估集，不同 step 检索索引不同。

## 9. Reward 设计细节

`P:agent_system/reward_manager/format_reward.py` + `episode.py`；配置 `ppo_trainer.yaml:259-294`。

- 最终分 = `1.0*outcome + 0.1*format`（`episode.py:332-339`），写在每条 row 最后一个 token。
- `format = (w_t*think_ok + w_a*action_ok + w_m*mem_ok)/(w_t+w_a+w_m)`，权重 1/1/1（`format_reward.py:162-167`）。
- `think_ok`：每步 ≥1 个 `<think>`（段长 ≥ `min_segment_chars=5`）；`action_ok`：每步恰好 1 个 `<action>`
  （`max_action_segments=1`，多则超限）（`:126-146`）。
- `mem_ok`：**整条轨迹**检索次数 ∈ `[1, max_memory_retrieve_segments=5]`（`:242-263`；yaml `:268`）；轨迹计数优先用环境侧
  `memory_retrieval_counts`，避免 XML 与实际检索不一致（`episode.py:217-223`）。
- 处罚：出现中文 → format 归零（`format_reward.py:121-124`，为 Qwen 中文漂移设的闸）；可选记忆超段数 → 硬归零（`:148-160`）。
  warmup：`format_warmup_global_steps` 内可乘系数、放宽 `require_memory_retrieve`/中文惩罚（`episode.py:182-214`）。
- **关键事实：格式奖励只奖励"有没有检索"，完全不奖励 critique/reconstruction 的质量。**
  改写质量的唯一（间接）信号是 episode return；针对重建的 shaping（与 P_old 完全相同惩罚、英文惩罚、下一步 reward 增量）
  在 `mem_adaptor_training.py:82-157`，**只在独立 adaptor GRPO 开启时生效**，发布脚本关闭（§10）。

## 10. 五阶段中哪些是"训练出来的"

| 阶段 | 需要 checkpoint？ | 依据 |
|---|---|---|
| observation | 否 | `env_manager.py:275-316` |
| retrieval（何时检索/query） | **是**：格式奖励 + cold-start 教格式 | `format_reward.py:242-263`；`scripts/build_alfworld_coldstart_data.py:466-480` |
| critique | **否（发布配方）**：只由 adaptor prompt 决定 | `ppo_trainer.yaml:610-611`；`mem_adaptor_rollout.py:68-101` |
| reconstruction | **否（发布配方）** | `train_alfworld.sh:255-258` |
| action | **是**：GRPO 主损失 | `train_alfworld.sh:218-254` |

决定性证据（`run_scripts/train_alfworld.sh:255-258`）：

```
mem_adaptor.enable=True
mem_adaptor.use_actor_rollout_wg=True    # 共享策略 worker
mem_adaptor.train_memory_adaptor=false   # 不训练 adaptor
mem_adaptor.model.path="${MODEL_PATH}"   # 与策略同权重
```

而 `train_memory_adaptor_enabled()` 要求 `use_actor_rollout_wg=false` **且** `train_memory_adaptor=true`
（`mem_adaptor_training.py:331-338`）。因此：

1. adaptor 与策略共享权重，但**是另一次独立生成**，其 token **不进 GRPO 损失**；
2. 仓库实现了"对 adaptor 的第二套 GRPO"（`mem_adaptor_training.py:387-467`：`episode_return_coef*R_episode + 英文惩罚 +
   下一步 reward 增量 − 与 P_old 完全相同惩罚`），但发布脚本未启用；
3. cold-start SFT 只教"消费 teacher 记忆 + 发检索 query"（`build_alfworld_coldstart_data.py:466-480`），**无一条 adaptor
   prompt 样本**（该 system prompt 全仓只在 `ppo_trainer.yaml:611` 出现）。

**【推测】** 论文的"GRPO 涌现改写"可能用了另一套配置（dedicated adaptor pool + `train_memory_adaptor=true`），
但仅凭发布物无法复现或验证。对无 GPU 的我们：**发布配方的 critique/reconstruction 本质是零样本 prompt 行为**。

## 11. 可复用组件（接口而非代码）

1. **`BaseMemoryStore` 抽象**（`P:agent_system/memory/memory_store.py:11-50`）：`initialize/add_records/retrieve/update_records/
   prune_low_utility_memories/count_records/sample_random_state_texts/close`；HTTP 与 Milvus 由 `MemoryStoreDispatcher` 切换
   （`:211-312`）。换后端不动上层。
2. **`MemoryEvent` 审计记录**（`types.py:57-66`）：每步落 `query_text/state_text/injected_text/retrieved[]`，并写进
   `non_tensor_batch`（`rollout_loop.py:487-504`）。现成的证据链形状。
3. **结构化记忆块 + 可逆替换**（`mem_adaptor_rollout.py:457-491`）：定位/替换/删除各一个函数，记忆可撤销、可审计、可对比。
4. **`retrieved_state_mode = matched|placeholder|random_vdb`**（`mem_adaptor_rollout.py:274-321`）：
   把 S_old 换成随机库内 state 做诱饵，**现成的反事实消融开关**，直接回答"改写是否真的用了源状态"。
5. **`global_step_schedule` 相位表**（`global_step_schedule.py:70-91`）：按 trainer step 切 `fixed→agentic`、
   控制 adaptor 的 env-step 窗口（`ppo_trainer.yaml:585-609`）。配置驱动的课程开关。
6. **`<EMPTY>` 协议 + 输出净化**（`mem_adaptor_rollout.py:68-101`）：一行正则挡掉"模型续写 agent 协议"，
   任何"一个模型兼做两件事"的系统都该抄。

不该照搬：FLAT+Milvus Lite、"检索即写库"、delete+reinsert 的 update、per-task collection。

## 12. 改造难点（无 GPU，只调 API）

需要替换的边界都在 `agent_system/`，`verl/` 可整体丢弃：

| 原实现 | 依赖 | 替换 |
|---|---|---|
| `actor_rollout_wg.generate_sequences`（策略/adaptor/summarizer） | vLLM+Ray+GPU | 任意 chat API；保留 `_build_adaptor_prompt`、`_normalize_adaptor_output`、`_splice_out_memory_block` 纯函数 |
| `build_adaptor_dataproto`/`DataProto` | verl | 删除，直接 messages→completion |
| `_summarize_batch_self` | actor worker | 改用已有 `_summarize_batch_teacher`（`:659-730`，OpenAI 兼容+重试+线程池），**这条路径本来就是 API 的** |
| `MilvusMemoryStore` | pymilvus/Lite | SQLite/pgvector/faiss；只需实现 `BaseMemoryStore` 七个方法 |
| `RemoteEmbeddingProvider` / `EpisodeRewardManager` | OpenAI 兼容 / torch | 前者原样可用（`:52-94`）；后者只取 `format_reward.py`（纯 Python） |
| `rollout_loop` 460-568 | verl | 重写成普通 `for step in range(max_steps)` |
| ALFWorld/WebShop env | 各自重依赖 | 文本环境适配层，保留 `build_text_obs → _pair_finalize_memory` 接缝 |

成本集中三处：(a) 第二次 generate 从 GPU worker 改 API（失去批量与共享 KV）；(b) batched rollout 改串行 loop（吞吐骤降）；
(c) utility 的"读即写"必须批量化/异步化。**不需要改**：prompt 文本、`<EMPTY>` 协议、注入格式、utility 公式、格式奖励。

## 13. stub / TODO / 硬编码 / 外部依赖

- 死代码：`_raw_trajectory_fallback_memory_record` 从未调用（`experience_summarizer.py:960-979`，注释自认 "Not used … future archival path"）；
  `memory_store.py:11-50` 全部 `NotImplementedError`；HTTP 后端不支持 export/rebuild（`:132-138`）。
- `environments/base.py:198`：基类 agentic 直接 `NotImplementedError`，各环境须自实现 `step_with_memory`。
- 硬编码：`milvus_memory.db`（`milvus_store.py:133`）、collection 前缀 `agent_memories_`（`:45-49`）、`bge_m3`/1024（`:59-60`）、
  `_TASK_RE="Your task is to:"`（ALFWorld 专用，`mem_adaptor_rollout.py:263`）、健康检查轮询（`local_service.py:227-252`）、
  teacher 默认 `api.openai.com/v1`+`gpt-4o-mini`（`ppo_trainer.yaml:446-452`）、脚本 embedding `localhost:8001`（`train_alfworld.sh:51`）。
- 硬依赖：embedding API、Milvus/FastAPI、vLLM（训练）、可选 Slurm；`requirements.txt` 含 `flash-attn`/`pymilvus`/`ray`，无 CPU-only 路径。
- 并发瓶颈：FastAPI 所有读写共用一把 `threading.RLock`（`memory_fastapi.py:19,35-124`），VDB 实际单线程。

## 14. 许可证：README 与 LICENSE 不一致

- **README badge 写 MIT**（`README.md:9`）；**LICENSE 是 Apache-2.0 全文**（`LICENSE:1-5`），模板占位未填
  （`Copyright [yyyy] [name of copyright owner]`）；`setup.py:92` 写 `license="Apache 2.0"`；`pyproject.toml:22` 指向 LICENSE。
- 源码头混杂：memory/env 基础文件是 NTU+verl-agent 的 Apache-2.0 头（`agent_system/memory/base.py:1-14`），而
  `experience_summarizer.py`、`mem_adaptor_rollout.py`、`memory_text_dedupe.py` 只写 "Copyright 2025 MemAdaptor"（无许可声明）；无 `NOTICE`。

**结论：实际许可是 Apache-2.0，MIT badge 是错的。** 含义：可以复用修改，但须保留声明、标注修改、满足 NOTICE 归属；
**不能**按 MIT 当作"随便摘一段"处理。EvoMemory 的 THIRD_PARTY 按 Apache-2.0 记录，并注明该不一致。

## 15. 对 EvoMemory 的启示

### 可吸收（具体）

1. **记忆块可定位、可替换、可删除**（`<memory>…</memory>` + splice）。每条经验带 `source_state/content/provenance`，
   注入整块替换、拒绝整块移除——"用了哪条、改成什么、为何删"全部进 trace，直接对应我们的四态区分。
2. **显式 reject 协议**：`<EMPTY>` 是零成本、可评测的"不适配"信号。把 accept-rate 当作负迁移抑制的一等指标，而非只看成功率。
3. **`MemoryEvent` 作一等审计对象**：每步 query/state/injected/retrieved，可直接搬进证据链，并与多模态证据（图像 URI、裁剪区域）并列。
4. **反事实开关**：`retrieved_state_mode` 的 random_vdb 诱饵，是检验"重建是否条件于源状态"的最小实验；我们的 A/B/C/D 应内建
   （真源状态 / 随机状态 / 无记忆）。
5. **utility 与存储解耦**：`(succ+1)/(use+2)` 放 metadata、由 manager 周期触发，换后端可用；需补 MemHarness 没有的
   **按证据来源分层统计**。**格式奖励是纯函数**（`format_reward.py` 零 torch），可直接用作 API 环境的协议检查器，
   但必须让 `require_memory_retrieve` 与记忆开关联动，别重蹈 §8 的对照污染。

### 不该照搬

1. **"检索即写库"的 utility 更新**：一次检索 k 次同步 update + delete/reinsert，API 环境下会放大成主要延迟。
   改为批量/异步/或 episode 结束一次性结算。
2. **FLAT + Milvus Lite + 每任务一个 collection**：单机实验配置，不是架构。EvoMemory 用 SQLite/Postgres + 可替换向量索引，
   更稳，也更容易做时间旅行与审计。
3. **"纯 GRPO 对照"**：已被格式奖励污染（§8）。baseline 必须显式关掉检索要求，并把"记忆开/关 × 有无重建"做成 2×2。

### "没有 GPU 就无法复现核心增益"——怎么诚实处理

- **代码事实**：发布配方里 critique/reconstruction 不进损失（`train_alfworld.sh:255-258` + `mem_adaptor_training.py:331-338`），
  真正被 GRPO 训练的只有 action 策略与检索时机。没有 GPU 损失的**不是**这套 pipeline，而是
  (a) "改写能力由训练涌现"；(b) 83.0% 的 latent guidance（关记忆仍强）——那必须靠 checkpoint。
- **建议表述**：我们复现的是 **harness 结构与记忆生命周期**（检索→判定→改写→注入→写回→剪枝），
  用 API 模型承担 critique/reconstruction；**不声称**复现 +8.8/+9.5，也不声称"能力来自 GRPO"。
- **建议实验**：A 无记忆；B replay（原文注入）；C reconstruct（我们的 prompt）；D reconstruct + 反事实 S_old。
  报告 accept-rate、重建与源状态一致性、OOD 成功率差。这是无 GPU 条件下**站得住**的贡献边界。
- **不要做**：把"用了 MemHarness 的 prompt"写成"复现了 MemHarness"；把 API 模型的重建质量归因于我们的训练；
  在报告里引用 85.2/75.6 当作自己的结果。

## 附：实际读到的文件

`agent_system/memory/` 全部 18 个 `.py`；`agent_system/environments/{base.py, env_manager.py, prompts/alfworld.py}`；
`agent_system/multi_turn_rollout/rollout_loop.py`；`agent_system/reward_manager/{format_reward.py, episode.py}`；
`verl/trainer/config/ppo_trainer.yaml`、`verl/trainer/main_ppo.py`；`run_scripts/` 三个脚本；
`scripts/{cold_start_sft.sh, build_alfworld_coldstart_data.py, extract_memory_records.py}`；
`README.md, LICENSE, setup.py, pyproject.toml, requirements.txt`。
