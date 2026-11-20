# 结构化可执行经验：设计

P1 交付物。回答 H1：**结构化经验表示能否提升复杂逻辑规则任务的成功率？**

---

## 1. 问题：一句话说不清"且"

上一阶段的实测结论（`docs/EXPERIMENT_RESULTS.md` §8）：

| family | 隐藏规则 | A | B | D |
|---|---|---|---|---|
| escalation | 排除 tier=gold，排除 note=hold（两个独立单属性） | 0/6 | 5/6 | 5/6 |
| conditional | 排除 **tier=gold 且 note=audit** | 0/6 | 1/6 | 0/6 |

D 的 5 次失败全是 `targets_actioned` 未完成（少做了 r04/r05），即**过度排除**：模型把
"exclude records with tier=gold **and** note=audit" 读成"...gold **or** ...audit"，
把只满足一半条件的诱饵记录也跳过了。

自然语言里，连接词只占一个词，但对行为是决定性的。**这不是提示词能修的问题**——
即使措辞再明确，读者仍然要自己解析逻辑结构。所以规则必须以数据形式存储、由代码求值。

---

## 2. 双表示

一条经验同时携带两种表示，各司其职：

| 表示 | 字段 | 用途 | 谁写 |
|---|---|---|---|
| 机器可执行 | `condition`（谓词 JSON） | 精确判定某条记录是否被规则排除 | 提取器提出，**代码校验后才保留** |
| 自然语言 | `lesson`（字符串） | 让模型理解"为什么"、支持跨领域迁移 | 模型或确定性模板 |

两者可以不一致，**冲突时以 `condition` 为准**，并记入 `experience_events`。
`condition` 为 `None` 的经验是纯散文经验，只能被读，不能被求值——这是诚实的降级，不是失败。

---

## 3. 谓词语言

```
node := { "all_of": [node, ...] }        # 合取
      | { "any_of": [node, ...] }        # 析取
      | { "not": node }                  # 否定
      | { "field": str, "op": op, "value": any }

op   := eq | ne | in | lt | lte | gt | gte | exists
```

约束（每一条都有测试）：

- **深度 ≤ 8，节点 ≤ 64**，防止膨胀与病态输入。
- **字段白名单**：求值时可传入允许字段集合，白名单外一律校验失败。
- **字段名不得含 `.`、`[`、`__`**，杜绝属性路径与 dunder 访问。
- **禁止 `eval` / `exec` / `getattr`**。谓词是数据，不是代码；LLM 生成的规则永远不能变成可执行程序。

### 3.1 三值逻辑

| 情形 | 结果 |
|---|---|
| 字段缺失（非 `exists`） | `UNKNOWN` |
| 类型不匹配（`gte` 遇到字符串） | `UNKNOWN` |
| `all_of`：任一 `FALSE` | `FALSE` |
| `all_of`：无 `FALSE` 但有 `UNKNOWN` | `UNKNOWN` |
| `any_of`：任一 `TRUE` | `TRUE` |
| `any_of`：无 `TRUE` 但有 `UNKNOWN` | `UNKNOWN` |

`UNKNOWN` **不得**被当作 `FALSE`。这是刻意的：把未知当假的系统会在字段缺失时静默停止执行规则，
而调用方永远不知道规则失效了。`Tri.__bool__` 直接抛 `TypeError`，从语言层面禁止 `if tri:`。

---

## 4. 连接词从反例中推断（本阶段的核心算法）

难点：从一次失败里，怎么知道规则是"且"还是"或"？

**答案在反例的结构里，不在措辞里。**

| 证据 | 推断 | 理由 |
|---|---|---|
| 同一条记录同时携带两个属性 | `all_of` | 它同时具备两项才被排除，说明两项都是必要条件 |
| 多条记录各携带**不同**的单属性 | `any_of` | 每一条单独就足以被排除 |
| 多条记录携带**相同**的单属性 | 单个叶子 | 只有一个判别属性 |

`predicate.from_counterexamples()` 实现这三条规则。它吃的是校验器输出的
`r03(tier='gold' and note='audit')` 这种**按记录分组**的反例，分组本身是关键信息——
把反例摊平成 `[gold, audit]` 就把这个信息丢了。

实测（`tests/test_harness.py::test_mode_S_executes_a_conjunction_that_prose_gets_wrong`）：

```
同一批 conditional 任务的第二个任务
  B（散文）  → failure （过度排除）
  S（可执行）→ success （5 次规则求值，正确区分诱饵）
```

---

## 5. 执行路径

```
Active Experience (condition)
        │
        ├── 注入给模型的散文：lesson
        │
        └── 注册为工具 check_rule(rid)
                 │
    Agent ── check_rule(rid) ──► 对每条 active 条件求值
                 │                 │
                 │                 ├─ TRUE     → exempt（这条记录该留着）
                 │                 ├─ FALSE    → 不在任何规则范围内
                 │                 └─ UNKNOWN  → 无法判定（字段缺失/类型不符）
                 ▼
           Agent 自行决定是否 set_field
```

关键点：

- **Harness 不替 agent 执行任务**。它只提供"求值"能力；要不要问、问了怎么用，由 agent 决定。
- **`exempt=null` 时保守处理**：脚本策略把"无法判定"当作"不排除"，并记入 trace；真实模型看到的是明确措辞。
- **规则来源可追**：返回里带 `experience_id` 与 `condition` 的文本渲染，事故可回溯到具体版本。

---

## 6. 提炼流程与准入门槛

```
Execution Trace + Outcome(program-verified)
        │
        ├── LLM 提出 {lesson, condition?}
        │        │
        │        └── condition 必须通过 schema + 白名单校验，否则丢弃
        │
        └── 确定性提取器（始终可用）
                 │  从 check detail 解析按记录分组的反例
                 └── from_counterexamples() → condition
                        │
                        ▼
              Candidate Experience
                        │
              （G/E 模式）程序化校验门
```

**单个成功案例不足以确立普遍规则**：成功路径的提取只产出散文（没有反例就没有条件），
因此成功经验永远不会生成可执行的排除规则。只有失败（被程序确认的"做错了"）才能产生条件。

---

## 7. 模式与消融设计

| 模式 | 表示 | Validation Gate | Critique | 用途 |
|---|---|---|---|---|
| A | 无记忆 | — | — | 下界 |
| B | 散文 | 无 | 无 | 上一阶段的朴素基线 |
| S | 双表示 | 无 | 无 | **H1**：结构化本身的效果 |
| G | 双表示 | 有 | 无 | **H4**：校验门的效果 |
| E | 双表示 | 有 | 选择性 | **H2**：条件化 critique 的效果 |
| D | 散文 | 有 | 全量 | 上一阶段的完整生命周期（对照） |

比较顺序按可解释性排：S−B 是表示的效果，G−S 是校验的效果，E−G 是 critique 的效果。

---

## 8. 有意不做的事

| 没做 | 原因 |
|---|---|
| 谓词编译成可调用的 Python 函数 | 会让 LLM 输出变成可执行代码，安全边界消失；当前规模下解释执行够快 |
| 支持 `sum/count` 等聚合谓词 | 需要引入记录集合语义，与本阶段的"单记录判定"模型冲突，留给后续 |
| 让 condition 直接驱动 `set_field` | Harness 会从"提供能力"退化成"替 agent 干活"，实验就失去意义 |
| 把散文与条件强制同步 | 两者本就服务于不同读者；冲突时以条件为准并留痕，比强行一致更可审计 |
