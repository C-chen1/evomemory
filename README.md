# EvoMemory

面向长程任务的多模态自演化经验记忆系统（Multimodal Experience Memory & Agent Harness）。

核心问题：Agent 如何从成功与失败的轨迹中提炼可复用经验，判断它是否适用于当前任务，并在经验出错时修改而不是污染未来任务。

---

## 快速开始

```bash
# 环境（Windows / Linux 通用；本项目零运行时第三方依赖）
uv venv
uv pip install -e ".[dev,images]"

# 全部测试（88 个）
PYTHONPATH=src python -m pytest -q

# 离线跑 A/B/C/D 四种模式（不需要 API key，不产生费用）
PYTHONPATH=src python -m evomemory.runner --modes ABCD --llm scripted --per-family 6

# 多模态记忆压缩研究（生成实测表格）
PYTHONPATH=src python scripts/compression_study.py
```

结果写到 `artifacts/run_<timestamp>/`：`results.json`、`summary.md`，以及每个模式独立的 `memory.db`。

### 接入真实模型

```bash
cp .env.example .env      # 填入 DEEPSEEK_API_KEY 与 DEEPSEEK_MODEL
PYTHONPATH=src python -m evomemory.runner --modes AD --llm api --per-family 6
```

代码不会猜模型 id：`model` 为空时直接报错。调用预算有硬上限（默认 200 次 / 50 万 token），超出即停止新请求。详见 `docs/NEXT_STEPS.md`。

---

## 设计要点

**模型只能提议，不能裁决。** 任务是否成功由程序比对环境状态决定；经验是否可信由 9 项程序化检查决定；经验能否被检索由状态机决定。模型的写权限被限制在"创建候选"这一步。

**经验按版本存行，内容不可变。** 修订产生 v(n+1)，旧版本标记 superseded 但永久可读——可以回答"这条经验被改过几次、每次为什么改、改之前是什么"。

**证据内容寻址。** 文件按 sha256 命名，写入后只读；缩略图与 caption 是独立证据行并带 `derived_from` 血缘，原图永不覆盖。

**结果可复现。** 相同配置两次运行逐位相同（有测试盯着）；离线实验使用确定性策略，结果文件里标注 `evidence_class=harness_selfcheck`，不会被误读为模型能力证据。

---

## 目录

```
src/evomemory/    14 个模块：models / db / evidence / memory / llm / retrieval /
                  lifecycle / harness / multimodal / policy / bench / metrics / runner / config
tests/            88 个测试
scripts/          多模态压缩研究
docs/             阶段文档 + 三份逐仓库审计
artifacts/        实验结果（含每个模式的 memory.db）
references/       三个参考仓库（git-ignored，commit SHA 见 docs/REPO_AUDIT.md）
```

---

## 文档索引

| 文档 | 内容 |
|---|---|
| `docs/FINAL_REPORT.md` | **总报告**：审计结论、复现结果、架构、测试、实验结果、简历口径 |
| `docs/REPO_AUDIT.md` | 三仓库横向审计结论 + 许可证矩阵 |
| `docs/_audit/*.md` | 逐仓库详细审计（file:line 级证据） |
| `docs/REPRODUCTION.md` | 实际执行过什么、没执行什么、为什么 |
| `docs/ARCHITECTURE_COMPARISON.md` | 三家架构并排比较 + 吸收矩阵 |
| `docs/INTEGRATION_DESIGN.md` | 六模块设计、接口、数据模型、职责边界 |
| `docs/DECISIONS.md` | 13 条设计决策（含被否方案与代价） |
| `docs/BENCHMARK_PLAN.md` | 实验设计、指标定义、效度威胁 |
| `docs/EXPERIMENT_RESULTS.md` | 实测数字与逐任务观察 |
| `docs/PROGRESS.md` | 当前进度、阻塞、下一步 |
| `docs/NEXT_STEPS.md` | 后续工作与解锁条件 |

---

## 状态

已实现并可运行：版本化经验记忆、内容寻址证据层、Harness 六机制、**三值谓词引擎与结构化双表示**、
**视觉面板基准与视觉输入链路**、A/B/C/D/S/G/E 八种实验臂、166 个测试。

已验证（真实模型、程序判分、配对检验）：记忆把成功率从 0% 提升到 75%（p=0.031）。

**已推翻的假设**：结构化表示与完整 critique 链路均未优于简单文本记忆（p=1.0，成本更高）。
原因已定位并写进 `docs/ABLATION_RESULTS.md`。未运行的实验（视觉 live、G 臂）在
`docs/PROGRESS.md` 的"当前阻塞"里列了可直接执行的命令。

详见 `docs/FINAL_REPORT.md`、`docs/RESUME_TECHNICAL_REPORT.md`。
