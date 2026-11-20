# 架构比较

比较对象：Recuris / MemHarness / CMA-Harness。三个仓库的完整证据见 `docs/_audit/`，交叉结论见 `docs/REPO_AUDIT.md`。本文只回答一个问题：**EvoMemory 的六个模块，每一个应该从谁身上学什么、明确不学什么。**

---

## 1. 三条生命周期并排

### Recuris（kernel 层，单轮）

```
真实 tool 回执 ──► _ground()            # 先用 receipt 结算上一轮，unmatched 只计数不伪造
                     │
      turn_start ──► Deliverer.select()  # 事件驱动的记忆投递
                     │
   WM manager.should_update() ──► _update_ledger()   # 模型只能 propose
                     │
              render_wm()            # 重编译上下文（不是追加历史）
                     │
              llm_draft() ──► Checker.inspect() ──► 至多一次 redraft
                     │
                commit() ──► pre_write deliverers ──► sanitizer（剥离伪造的 trusted block）
```

关键性质：**上下文是每轮重编译的，不是累积的**。这是它和"聊天历史 + 检索拼接"最本质的区别，也是"状态驱动检索"能成立的原因。

### MemHarness（单 episode）

```
reset ──► build_text_obs
            │
   fixed:   build_memory_message(S_curr) ──► 注入 <memory> 块
   agentic: append_retrieval_hint ──► 模型自己发 <memory_retrieve> ──► 检索 ──► 注入
            │
   generate ──► 再跑一次 adaptor 生成 ──► <EMPTY>? 删块 : 替换块内容
            │
        env.step ──► 循环
            │
   episode 结束 ──► summarizer 写回 ──► utility 更新 ──► 周期剪枝
```

关键性质：**记忆块是可定位、可整体替换、可整体删除的容器**。这让"用了哪条、改成什么、为什么删"全部可审计。

### CMA-Harness（单 turn）

```
用户指令 ──► (可选) VLM B 检索全量记忆库 ──► rank 列表
              │
        VLM C 改写为 edit/generate prompt
              │
        Qwen-Image(-Edit) 生成
              │
        VLM A 对新图摘要 ──► 追加进 memory_bank
```

关键性质：**每张产物立刻写回记忆**，因此记忆单调膨胀；`diff` 是自然语言，没有 `parent_id`，编辑谱系不可追溯。

---

## 2. 六个模块的吸收矩阵

| EvoMemory 模块 | 主要来源 | 具体吸收什么 | 明确不吸收 |
|---|---|---|---|
| **1. Execution Harness** | Recuris | `HarnessPort` 式接缝（agent 与 kernel 之间约 12 个方法）；"每轮重编译上下文"；写权限三通道 | 11k 行的 `metaagent/driver.py`；tau2/Claude Code/Docker 依赖链 |
| **2. Multimodal Evidence Store** | CMA + 自研 | "检索表示 / 执行证据"分离（缩略图检索、原图生成）；原始证据永不覆盖 | CMA 的全部代码（无授权）；`align_image_to_model` 原地 resize 覆盖 |
| **3. Experience Memory** | Recuris + 自研 | 卡片式经验；`source`/`trigger` 字段；包级不可变版本 | 只支持 `str` 的 `EMEntry.body`；没有 superseded 状态这一点 |
| **4. Retrieval & Reconstruction** | MemHarness | `<memory>…</memory>` 可逆块；`<EMPTY>` 显式拒绝；`MemoryEvent` 审计记录；`retrieved_state_mode` 反事实开关 | Milvus FLAT + 每任务一个 collection；"检索即写库"的 utility 更新 |
| **5. Evolution & Validation** | Recuris | 对 **item** 重采样的配对 bootstrap；`reg_cap`；power gate / MDE；"弃权是合法结果"（O/P 双变量） | 把 gate 当唯一真理（它先在 CI 上拒了 reg_cap 想放的 case）；`settle.py` 的 20-cell 冻结算术 |
| **6. Benchmark & Evaluation** | 三家 + 自研 | 同一函数生成两个 arm、只切换一个变量；按会话切分防泄漏；机制指纹先于分数 | "同预算"只对齐 task×trial（Recuris）；被污染的纯 GRPO 对照（MemHarness） |

---

## 3. 三处必须自己做的设计（三家都没有）

### 3.1 卡片级版本与显式血缘

三家的版本策略分别是：Recuris = 包级目录不可变（卡片无版本，淘汰只能删除）；MemHarness = 记录不可变、utility 可变；CMA = 无版本、无 `parent_id`。

三者都回答不了这个问题：**"这条经验被改过几次，每次为什么改，改之前是什么样？"**

EvoMemory 的做法：`Experience` 有 `experience_id`（稳定）与 `version`（递增），每次修改产生一条新的不可变版本行，旧行标记 `superseded` 并写入 `supersedes` 边。查询默认只返回 `active` 版本；任何历史版本可回滚。

### 3.2 证据内容寻址

Recuris 的 embedding 缓存键只含 doc id 不含正文（`builtin/embed_retrieval.py:89-103`）；CMA 的缩略图缓存键 `md5(image_path)[:12]` 不含 mtime/size。两者都会在内容变更后命中陈旧缓存。

EvoMemory：所有证据文件按 `sha256(bytes)` 命名并做去重存储；证据一旦写入只读；缩略图/派生表示作为**独立证据**记录 `derived_from` 边，而不是覆盖原图。

### 3.3 预算可计量

Recuris 的"同预算"只成立于 task×trial；MemHarness 的 baseline 被格式奖励污染；CMA 完全没有 token 计量。

EvoMemory：`LLMBudget` 强制记录每次调用的 input/output tokens、调用数、tool 调用数，并在 run 级汇总。A/B/C/D 四个模式的报告必须并列实际消耗，预算不对齐的结果**不允许**作为有效性证据。

---

## 4. 允许/禁止的行为对照

| 行为 | Recuris | MemHarness | CMA | EvoMemory |
|---|---|---|---|---|
| 模型自己宣布任务完成 | 禁止（`WritePermissionError`） | 无此概念 | 无此概念 | **禁止**，只能由 harness 依据验收结果写 |
| 模型自己宣布某条记忆"已验证" | 禁止 | 无此概念 | 无此概念 | **禁止**，`verification_status` 只能由 validator 写 |
| 模型改写记忆内容 | 允许（propose），需 gate 通过 | 允许（每次推理都改写） | 允许（写回摘要） | 允许 propose，需 validator + gate 通过 |
| 检索不到时的行为 | 投递空 / 走 fallback | 注入中性提示 | 走 generate 分支 | 注入中性提示并记 `retrieval_miss` 事件 |
| 记忆删除 | `remove_card`（硬删） | 剪枝（utility 阈值） | 无 | 软删（`deprecated` + 原因 + 证据），可回滚 |

---

## 5. 为什么不做成"复刻任一仓库"

如果以 Recuris 为骨架：会继承 11k 行 driver 与 tau2/Docker/Claude Code 的耦合，且它的文档与代码已经在多处不一致——复刻等于继承这些缺口。

如果以 MemHarness 为骨架：核心能力（改写质量）在发布配方里没有训练信号，等于复刻一个"结构对、训练证据缺失"的系统；而它的增益数字无法在当前环境下验证。

如果以 CMA 为骨架：**法律上不允许**（无授权）。

因此 EvoMemory 的定位是：**独立实现的第六个系统，把三家已验证有效的局部机制（写权限、配对 gate、可逆记忆块、证据分离、反事实开关）组合到同一个数据模型上，并对每个机制配一个可执行断言。**

---

## 6. 三个仓库各自最有价值的一行

- Recuris：`accept = lo > 0 and n_dn <= reg_cap`——**准入是算术，不是模型投票**。
- MemHarness：`<EMPTY>`——**拒绝是最便宜、最该被计量的动作**。
- CMA：检索用缩略图、生成用原图——**"检索表示"和"执行证据"可以是两样东西**。
