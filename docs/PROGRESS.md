# 进度记录

最后更新：2026-11-21（会话内持续推进）

---

## 当前阶段

**第一阶段（源码审计 + 最小复现）、第二阶段（架构设计）、第三阶段（核心闭环）、第四阶段（多模态）、第五阶段（Harness）已完成；第六阶段（Benchmark）离线部分已完成，真实模型实验被 API key 阻塞。**

---

## 已完成

| 模块 | 文件 | 状态 | 证据 |
|---|---|---|---|
| 三仓库克隆与元数据 | `references/` | 完成 | `docs/REPO_AUDIT.md` 表 1（commit SHA / 许可证） |
| 源码审计 | `docs/_audit/*.md` | 完成 | 三份 320-350 行、带 file:line 的审计 |
| 架构比较 | `docs/ARCHITECTURE_COMPARISON.md` | 完成 | —— |
| 最小复现 | `docs/REPRODUCTION.md` | 完成 | CMA 14/14 单测通过；Recuris gate 与 MemHarness 去重的行为探针 |
| 统一架构 | `docs/INTEGRATION_DESIGN.md` | 完成 | —— |
| 决策记录 | `docs/DECISIONS.md` | 完成 | 13 条，含被否方案与代价 |
| 数据模型 | `src/evomemory/models.py` | 完成 | 6 个枚举、4 条状态机约束 |
| 持久化 | `src/evomemory/db.py` | 完成 | 版本化 schema，9 张表 |
| 证据存储 | `src/evomemory/evidence.py` | 完成 | 内容寻址 + `derived_from` |
| 版本化记忆 | `src/evomemory/memory.py` | 完成 | 状态机 + 追加式事件日志 |
| LLM 与预算 | `src/evomemory/llm.py` | 完成 | 调用前检查硬上限 |
| 检索 | `src/evomemory/retrieval.py` | 完成 | BM25 + 哈希向量 + RRF |
| 生命周期 | `src/evomemory/lifecycle.py` | 完成 | 提取 / 校验 / critique / 重建 |
| Harness | `src/evomemory/harness.py` | 完成 | 六项机制全部落地 |
| 多模态 | `src/evomemory/multimodal.py` | 完成 | 附件、感知、压缩实测 |
| 离线策略 | `src/evomemory/policy.py` | 完成 | 确定性规则跟随器 |
| Benchmark | `src/evomemory/bench.py` | 完成 | 程序化验收，无 LLM judge |
| 指标 | `src/evomemory/metrics.py` | 完成 | 6 项指标 + MDE |
| 运行器 | `src/evomemory/runner.py` | 完成 | A/B/C/D，每模式独立 DB |
| 测试 | `tests/` | 完成 | **88 个测试全绿** |
| 压缩研究 | `scripts/compression_study.py` | 完成 | 48 个实测单元 |

### 已通过的测试（88）

```
tests/test_bench.py          11
tests/test_determinism.py     3
tests/test_evidence.py        7
tests/test_harness.py        12
tests/test_lifecycle.py      13
tests/test_llm_retrieval.py  12
tests/test_memory.py         11
tests/test_multimodal.py     12
tests/test_replay_recovery.py 7
```

运行方式：`PYTHONPATH=src python3 -m pytest -q`

### 当前实验结果（离线，`harness_selfcheck`）

| 模式 | 成功率 | 负迁移率 | token/任务 | 准入经验 |
|---|---|---|---|---|
| A | 0.0% (0/12) | 0.0% | 4232 | 0 |
| B | 58.3% (7/12) | 25.0% | 3813 | 12 |
| C | 58.3% (7/12) | 25.0% | 3813 | 12 |
| D | 83.3% (10/12) | 0.0% | 3399 | 4 |

详见 `docs/EXPERIMENT_RESULTS.md`。**不得作为"记忆提升大模型能力"的证据引用。**

---

## 当前阻塞

1. **没有 DeepSeek API key**。`.env` 尚未创建（已提供 `.env.example`）。所有 live-model 实验无法执行。
   - 影响：无法确认 provider 实际支持的模型 id；无法跑真实模型的 A/B/C/D。
   - 处理：离线管线已完整可用；待 key 就位后按 `docs/NEXT_STEPS.md` 第 1 节执行。
2. **没有 GPU / 无法访问 Windows 环境**。视觉语言模型与视觉嵌入无法运行，`C` 模式在离线条件下退化为 `B`。
   - 影响：多模态**效果**未验证（存储、检索、压缩**机制**已实现并测试）。
3. **参考仓库的重型依赖无法安装**：Docker（SkillFlow/TB2.1）、vLLM（MemHarness 训练）、8 卡（CMA）。已在 `docs/REPRODUCTION.md` 第 3 节逐项记录为"未执行"。

---

## 环境边界（重要）

本会话的 shell 是 Linux 沙箱（2 vCPU / 3.8 GiB / 无 GPU），**不是用户的 Windows 主机**。因此：

- 无法探测 Windows 侧的 Python、uv、Docker、显卡状态；
- 无法读取 Windows 系统环境变量（包括可能已配置的 API key）；
- `.venv` 建在沙箱内，D 盘只有源码（按用户选择）。

用户在自己的 Windows 上运行需要：`cd 06_evomemory && uv venv && uv pip install -e ".[dev,images]"`，然后 `python -m evomemory.runner`。

---

## 最近修改文件

```
src/evomemory/{models,db,evidence,memory,llm,retrieval,lifecycle,harness,policy,bench,metrics,runner,config,multimodal}.py
tests/*.py                       (9 个文件, 88 测试)
scripts/compression_study.py
docs/{REPO_AUDIT,ARCHITECTURE_COMPARISON,REPRODUCTION,INTEGRATION_DESIGN,DECISIONS,BENCHMARK_PLAN,EXPERIMENT_RESULTS,PROGRESS,NEXT_STEPS}.md
docs/_audit/{recuris,memharness,cma_harness}.md
artifacts/run_<ts>/               (4 个模式的 db + results.json + summary.md)
artifacts/multimodal/             (compression_study.json + REPORT.md)
```

---

## API 使用量

| 项 | 值 |
|---|---|
| 调用次数 | **0 / 200** |
| token 消耗 | **0 / 500,000** |
| 原因 | 无 key（`Settings.dry_run = true`）；所有离线运行走确定性脚本客户端，不计费 |

`llm_calls` 表已在每次调用时无条件写入（即使 `db=None` 之外的路径都记录），key 就位后可直接从数据库核对用量，不依赖人工统计。

---

## 下一步

1. 等 `.env` → 确认模型 id → 跑 live A/B/C/D（12 任务，约 100-150 次调用）。
2. 视觉侧：接入任一 VLM 实现 `Perceiver`，让 `C` 模式真正不同于 `B`。
3. 扩大任务集到 50-100 个 cell，把 MDE 从 ±57pp 降到 ±15pp 量级。
4. 把统计 gate 接进批次级演化（修订 / 废弃经验的决策），而不是只做单条校验。

详见 `docs/NEXT_STEPS.md`。
