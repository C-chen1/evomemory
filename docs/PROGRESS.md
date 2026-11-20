# 进度记录

最后更新：2026-11-21（第二阶段 P0–P3 完成，P4/P5 文档完成）

---

## 阶段状态

| 阶段 | 状态 | 关键产出 |
|---|---|---|
| P0 基线复核与安全 | ✅ 完成 | `BASELINE_AUDIT.md`、`SECURITY_AUDIT.md`；密钥从未进入 git 历史（实测） |
| P1 结构化可执行经验 | ✅ 完成 | `predicate.py`、双表示经验、反例连接词推断、`check_rule` 工具 |
| P2 经验演化与选择性 critique | ✅ 完成（E 未跑满） | 消融 B/S/E、`ABLATION_RESULTS.md`、`FAILURE_ANALYSIS.md` |
| P3 多模态视觉记忆 | ⚠️ 代码完成，**live 未运行** | `visual_bench.py`、视觉客户端、V0–V3 实验脚本、泄漏防护 |
| P4 统一 Benchmark | ✅ 协议与统计完成 | `EXPERIMENT_PROTOCOL.md`、聚类分析、有效独立性报告 |
| P5 工程强化与交付 | ✅ 完成 | 166 测试、`RESUME_TECHNICAL_REPORT.md` |

---

## 本阶段最重要的三个结论

1. **H1 未获支持**：结构化记忆与散文记忆在 8 个任务上完全打平（6/8 vs 6/8，p=1.0），
   结构化多花 43% token。原以为的"散文表达不了合取"被推翻——真正的问题是经验文本缺操作数。
2. **H2 的动机被数据证实但效果未验证**：18 任务里 38 次 critique 有 35 次判定 apply/adapt，
   条件化后调用降到 0；但 n=5，无法判断是否无损。
3. **H3/H4 无数据**：视觉 live 未跑；校验门缺少"记忆会变坏"的任务。

---

## 测试与提交

- 测试：**166 个全绿**（`PYTHONPATH=src python3 -m pytest -q`）
- 起始 commit：`96ff181`
- 本阶段提交：`2944f1e`(P1) → `116c9d8`/`1406fa0`(P2) → `170d7a7`(P3) → 本次

---

## API 用量

| 项 | 值 |
|---|---|
| 本轮调用 | **449 / 450**（上限已到，停止新请求） |
| 本轮 token | 约 62 万 / 2,000,000 |
| 明细 | 视觉冒烟 3；消融 A 98 / B 91 / S 96 / E 51；作废的 S 首轮 113 |
| 累计（两阶段） | 约 1,958 次调用 |

作废成本：结构化首轮 113 次调用因步数预算缺陷作废（`artifacts/ablation_stepbug/`），如实记录。

---

## 当前阻塞

1. **API 调用上限已到**：P3 的 live 实验需要新的预算额度。命令已就绪（见下）。
2. **G 模式（结构化 + 门，无 critique）未运行**：H4 无法回答。
3. **E 只跑了 5 个任务**：无法判断"跳过 critique 是否无损"。

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
