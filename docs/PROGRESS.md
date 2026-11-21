# 进度记录

## 第三阶段接手复核（2026-11-21）

### 离线组件进展

- 新增面板专用回放准入入口：从已存训练反馈及原图重建红灯作用域的样本，记录验证报告；反例拒绝，覆盖不足保持 pending。已覆盖错误琥珀规则的端到端构造测试。旧实验臂不改变；视觉实验脚本尚未使用新入口。版本更新仍保持 pending，待实现安全版本切换与归因回归。
- `panel-v2` 新生成器打乱区位类别并隐藏种子，旧生成器与历史数据保留；跨平台字体差异使图像字节未必与历史实验一致。训练/测试拆分尚需由实验入口强制实施。
- 结构化反馈要求完成状态的任务和运行 trace、可核验的匹配图像与程序评分，并核对数据库中的 trace 与判分记录；这不是防篡改数据库的独立证明。面板准入要求展示文本由谓词确定性生成，防止规则通过而矛盾的自由文本被注入。
- 完整测试 170 项通过；仍未发起 API 调用。

- 新增训练任务的结构化视觉评分反馈：区分 Agent 观察、模型解释、评分真值和动作结果，关联任务、运行和图像证据；拒绝测试集输入。
- 新增独立的结构化候选规则回放检查：检查谓词、证据、family、版本、动作及正反例；对未知值或覆盖不足返回 pending，对已知冲突返回反例。它目前**尚未接入记忆准入或视觉实验**，也不构成完整的冲突与历史回归检查。
- Windows 完整测试 168 项通过。未运行 API 实验；后续仍需强制执行完成后反馈边界、验证证据与样本绑定、在 gate 强制准入、扩展冲突和版本回归，并完成统一预算、数据隔离和实验。

- 起点 `46d43a4`，工作区干净。Windows 默认 Python 缺 pytest；Anaconda Python 原始基线为 153 通过、13 失败，原因是视觉渲染硬编码 Linux 字体路径。同款 DejaVu 字体临时替代后 166 项通过；增加跨平台字体回退后直接运行同样 166 项通过。
- 视觉任务固定把目标、封锁和非目标放在 A、B、C，存在位置标签泄漏风险，不能以既有防护测试证明不存在位置捷径。历史任务和结果未改动。
- 四个历史视觉数据库合计 32 条图像证据的现存文件均通过 SHA-256 核验；但证据元数据缺运行 ID，且同内容复用首条元数据，不能证明完整任务与运行来源链。
- 视觉实验脚本按臂建立独立预算，且已有预算账本对部分失败响应记零 token、请求后才检查 token 上限。本轮尚未进行付费调用；需要统一账本和上限控制后才能启动实验。
- 尚未完成：结构化反馈、反例验证及其接线、演化回归、V2-new、G 和扩样实验。历史结果仅作参考，不与新数据假定严格配对。

最后更新：2026-11-21（第二阶段 P0–P3 完成，P4/P5 文档完成）

---

## 阶段状态

| 阶段 | 状态 | 关键产出 |
|---|---|---|
| P0 基线复核与安全 | ✅ 完成 | `BASELINE_AUDIT.md`、`SECURITY_AUDIT.md`；密钥从未进入 git 历史（实测） |
| P1 结构化可执行经验 | ✅ 完成 | `predicate.py`、双表示经验、反例连接词推断、`check_rule` 工具 |
| P2 经验演化与选择性 critique | ✅ 完成（E 未跑满） | 消融 B/S/E、`ABLATION_RESULTS.md`、`FAILURE_ANALYSIS.md` |
| P3 多模态视觉记忆 | ✅ 完成（V0/V1/V2 已跑） | `visual_bench.py`、视觉客户端、V0–V3 实验脚本、泄漏防护、实测结果 |
| P4 统一 Benchmark | ✅ 协议与统计完成 | `EXPERIMENT_PROTOCOL.md`、聚类分析、有效独立性报告 |
| P5 工程强化与交付 | ✅ 完成 | 166 测试、`RESUME_TECHNICAL_REPORT.md` |

---

## 本阶段最重要的五个结论

1. **H1 未获支持**：结构化记忆与散文记忆在 8 个任务上完全打平（6/8 vs 6/8，p=1.0），
   结构化多花 43% token。原以为的"散文表达不了合取"被推翻——真正的问题是经验文本缺操作数。
2. **H2 的动机被数据证实但效果未验证**：18 任务里 38 次 critique 有 35 次判定 apply/adapt，
   条件化后调用降到 0；但 n=5，无法判断是否无损。
3. **H3 未获支持，但视觉链路已验证**：V0（无图）targets 0/8、V1（有图）targets 8/8 但
   lockdown 0/8 —— 感知正确，缺的是规则。V2 的失败驱动学习**学出了错误规则**（把 lockdown
   归因成"琥珀灯"），成功率与 V1 相同。
4. **演化机制是本阶段唯一正向结果**：规则漂移后，归因式演化把 2/8 提到 5/8
   （漂移后 0/5 → 3/5），新增 3 个成功、丢失 0 个；McNemar p=0.25，n=8 下不显著。
5. **关联式退休被实测否决**：它会连正确规则一起退休，导致 agent 失去全部记忆。

---

## 测试与提交

- 测试：**166 个全绿**（`PYTHONPATH=src python3 -m pytest -q`）
- 起始 commit：`96ff181`
- 本阶段提交：`2944f1e`(P1) → `116c9d8`/`1406fa0`(P2) → `170d7a7`(P3) → 本次

---

## API 用量

| 项 | 值 |
|---|---|
| 本轮调用（第三轮） | **约 533**（无新设上限，用户指示继续） |
| 本轮 token | 约 60 万 |
| 明细 | 漂移实验 353（含被否决的第一版机制 177）；视觉 134 + 46（未落库） |
| 累计（两阶段） | 约 1,958 次调用 |

作废成本：结构化首轮 113 次调用因步数预算缺陷作废（`artifacts/ablation_stepbug/`），如实记录。

---

## 当前阻塞

1. **G 模式（结构化 + 门，无 critique）仍未运行**：H4 无法回答。
2. **演化实验 n=8，p=0.25**：需要约 20 个漂移后任务才能确认。
3. **视觉 V2 学出错误规则**：失败归因需要程序化操作数 + 回放验证，见
   `docs/MULTIMODAL_MEMORY_DESIGN.md` §9.4。

解除方式：

```bash
PYTHONPATH=src python scripts/vision_smoke.py                       # 3 次调用
PYTHONPATH=src python scripts/vision_experiment.py --modes V0,V1 --tasks 8 --max-calls 120
PYTHONPATH=src python -m evomemory.runner --modes G --llm api --per-family 4 \
  --families escalation,conditional --run-dir artifacts/ablation --max-calls 120
```

---

## 最近修改文件

```
src/evomemory/predicate.py        新增：三值谓词引擎 + 反例连接词推断
src/evomemory/visual_bench.py     新增：视觉面板环境（无文本泄漏）
src/evomemory/harness.py          check_rule 工具（批量）、结构化步数额外、选择性 critique 接线
src/evomemory/lifecycle.py        _cites_operand、_violation_attrs、MemoryCritic(selective)
src/evomemory/llm.py              images 参数（data URL 内联）、空响应守卫
src/evomemory/metrics.py          mcnemar_exact、paired_bootstrap_ci、by_cluster、effective_independence
src/evomemory/policy.py           可执行路径（批量 check_rule）
scripts/vision_smoke.py           视觉能力冒烟（已跑通：3/3 通过）
scripts/vision_experiment.py      V0–V3 实验脚本（dry-run 已验证链路）
tests/                            166 个测试
docs/                             14 份文档（见下）
```

---

## 文档索引

`BASELINE_AUDIT` / `SECURITY_AUDIT` / `STRUCTURED_MEMORY_DESIGN` / `MULTIMODAL_MEMORY_DESIGN` /
`EXPERIMENT_PROTOCOL` / `ABLATION_RESULTS` / `FAILURE_ANALYSIS` / `DECISIONS` / `PROGRESS` /
`NEXT_STEPS` / `FINAL_REPORT` / `RESUME_TECHNICAL_REPORT` / `REPO_AUDIT` / `REPRODUCTION`。

---

## 下一步（按证据价值排序）

1. 跑 `G` 与跑满 `E`：这是唯一能把 H2/H4 从"机制成立"推进到"效果成立"的动作。
2. 构造**会变坏的任务**（过期规则、近似错误规则、冲突经验）——校验门只在这类任务上可能有独立贡献。
3. 用**更弱的模型**复测 S vs B：若散文在弱模型上重新失效而结构化仍稳，
   才说明结构化的价值在"能力下界"而不是平均情况。
4. 跑视觉 V0/V1（120 次调用）验证环境与链路，再跑 V2/V3。
