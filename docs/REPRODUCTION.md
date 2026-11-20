# 最小复现报告

本文件只记录**实际执行过并拿到输出**的事情。没跑过的写"未执行"，不写"应该可以"。

执行环境：Linux 沙箱（`uname` = `Linux 6.8.0-138-generic x86_64`），Python 3.10.12，2 vCPU / 3.8 GiB RAM / **无 GPU（无 `nvidia-smi`）**。
注意：这不是用户的 Windows 主机。Windows 侧的 Python/uv/Docker/显卡状态**本会话无法探测**，见 `docs/PROGRESS.md` 的"环境边界"。

---

## 1. 仓库获取

| 仓库 | 结果 | 方式 | 备注 |
|---|---|---|---|
| Recuris | ✅ 完整 | `git clone` | 4.0 MB，210 tracked files，72 `.py` |
| MemHarness | ✅ 完整 | `git clone --depth 1` | 38 MB，491 `.py`（含 verl fork） |
| CMA-Harness | ⚠️ **部分** | `git fetch --depth 1 --filter=blob:none` + `git cat-file` | 见下 |

CMA-Harness 的完整 clone 失败 4 次，有两种失败模式：

1. `git clone` 与 `git fetch`（非 filter）在 ~45-150 s 后 `fetch-pack: unexpected disconnect while reading sideband packet`，退出码 124/128。仓库含 8 个 mp4 与多份 pdf（`assets/demos/`、`assets/*.pdf`），推测是包体积 + 链路不稳。
2. 一次性残留 `.git/shallow.lock`，导致后续 `git fetch` 报 "Another git process seems to be running"。清理方式：`find .git -name "*.lock" -delete`。

成功路径：`git fetch --depth 1 --filter=blob:none origin main` 只取 tree（79 个路径），随后用 `git ls-tree -r FETCH_HEAD` 得到 blob SHA，对 `research/` 下 38 个文本文件逐个 `git cat-file -p <sha>` 抽取。工作区仍为空，`.git` 保留，`FETCH_HEAD` = `84732bcd6e838d1917d17184b98d3e416e4325c9`。

抽取结果保存在 `references/cma-harness-files/`，文件名是原路径把 `/` 换成 `_`；原始路径清单在 `references/cma-harness-files/_BLOB_MANIFEST.txt`。**未获取到的文件（mp4/pdf/图片）没有伪造，也没有影响任何代码级结论。**

---

## 2. 实际跑通的复现

三条都拿到了真实输出，命令可重放。

### 2.1 CMA-Harness 的 CPU 单测：14/14 通过

作者在 `research/README.md` 里说这套测试是 CPU-only 的发布检查。为了跑它，先把 38 个扁平文件按 `_BLOB_MANIFEST.txt` 还原成原始目录结构（`research/benchmark/`、`research/ms-swift/`、`research/tests/` 等），共 38 个文件，无缺失。

```bash
python3 -I -m unittest discover -s research/tests -v
```

输出（尾部）：

```
test_api_constructor_keywords_match_service ... ok
test_baseline_discovers_exported_service ... ok
test_legacy_evaluation_json_parser ... ok
test_score_parser_boundaries ... ok
test_judge_returns_string ... ok
test_missing_endpoint_fails_before_network ... ok
test_mocked_request_and_text_response ... ok
test_response_blocks_and_image_inputs ... ok
test_scenario_image_adapter_reads_generated_file ... ok
test_null_prediction_and_skipped_target ... ok
test_sorted_exact_match_and_range ... ok
test_empty_and_exact ... ok
test_normalize_text_and_empty_overlap ... ok
test_partial_and_hard_weight ... ok
----------------------------------------------------------------------
Ran 14 tests in 0.044s
OK
```

**结论**：这份快照在无 GPU 环境下确实能自检通过。同时确认了它的测试覆盖面——奖励算术、检索指标边界、HTTP 适配器的请求形状。它**不覆盖**任何记忆压缩、检索质量或生成质量（那些都需要模型）。

运行安全性检查（执行第三方代码前做的）：`research/tests/test_release.py` 只 import `ast/importlib/json/sys/tempfile/types/unittest/pathlib/mock`，所有 `requests.post` 都被 `patch.object` 替换，写操作只落在 `tempfile.TemporaryDirectory()`。无网络、无破坏性副作用，因此可执行。

### 2.2 Recuris 的 `held_out_paired_gate`：行为符合文档，且比文档更严

`src/recuris/metaagent/gates.py` 只依赖 `random / statistics / dataclasses`，可直接 import。用合成数据探针（`work/repro/repro_probe.py`）跑了四种情形（12 个 item，每 item 4 个 seed）：

```
uniform +0.5 : accept=True  net=+0.5000 ci=(+0.5000,+0.5000) up=12 dn=0
identical    : accept=False net=+0.0000 ci=(+0.0000,+0.0000) reason='CI includes 0: not significant'
mixed net>0  : accept=False net=+0.5000 up=9 dn=3 reason='CI includes 0: not significant'
deterministic: True
```

三点值得记录：

- **拒绝是默认行为**。相同 arm 被拒（CI 含 0），这符合"不确定就不采纳"。
- 第三种情形我原本想验证 `reg_cap`，结果它**先被 CI 拒了**：9 个 item 各 +1、3 个 item 各 −1，净 +0.5，但对 item 重采样后 CI 跨 0。这说明这个 gate 对"少数 item 大幅回归"的容忍度比 `reg_cap` 参数暗示的更低——**门槛实际上是 CI 和 cap 的联合，不是后者**。
- 常数差值会让 bootstrap CI 退化成零宽度（`(+0.5,+0.5)`）。EvoMemory 自己实现时必须显式处理零方差，否则"全 item 完全一致"这种退化输入会给出过度自信的区间。

这个探针给 EvoMemory 的准入 gate 提供了一个**可对照的黄金参考**。

### 2.3 MemHarness 的去重纯函数：行为符合注释

`agent_system/memory/memory_text_dedupe.py` 零依赖。探针输出：

```
threshold .95 -> kept=[0, 2, 3]   # keep-first greedy，第 1 项被判为与第 0 项近重复
threshold 0.0 -> kept=[0, 1, 2, 3] # threshold<=0 时禁用去重
cosine([1,0],[0,1])=0.000
cosine len-mismatch=0.000
```

符合函数 docstring。注意**长度不匹配时静默返回 0.0**（不是抛异常），这意味着一个 bug 可以伪装成"完全不相似"从而让去重失效而不报错。EvoMemory 的相似度函数在长度不匹配时必须显式抛错。

---

## 3. 明确未执行的部分（以及为什么）

| 项目 | 状态 | 阻塞原因 |
|---|---|---|
| Recuris `recuris tau2` 任何运行 | **未执行** | 需要 tau2-bench checkout + 在线 LLM user simulator/judge + vLLM 或 frontier endpoint |
| Recuris `recuris metaagent run`（演化 campaign） | **未执行** | 需要 Docker + Compose V2 + Claude Code CLI + Anthropic→OpenAI proxy |
| Recuris `recuris tta run` | **未执行** | 需要 harbor + Docker 镜像 |
| MemHarness 任何训练/评测 | **未执行** | 需要 8 GPU + vLLM + Ray + flash-attn；`requirements.txt` 无 CPU-only 路径 |
| MemHarness ALFWorld/WebShop | **未执行** | 环境安装需下载游戏资源与 MaskRCNN 权重，且需 GPU 推理 |
| CMA 任何 benchmark / 服务 | **未执行** | 8 卡角色硬编码；`gpu_count == 0` 直接 `RuntimeError('No available GPU; model cannot run on CPU')`；且权重与 M2CA-Bench 未发布 |
| CMA VLM A/B 的 SFT/RL | **未执行** | 8 卡 + ZeRO-2 + flash-attn + 独立奖励服务进程 |

**因此本项目的复现结论只到"代码可读、可静态核对、纯函数级可执行"这一层。** 任何"复现了论文数字"的说法都是不成立的，本项目也不会这么写。

---

## 4. 尝试过的补救路径与结果

| 尝试 | 结果 |
|---|---|
| CMA 完整 `git clone` ×2 | 超时/断连，无工作区 |
| CMA `git fetch --depth 1` ×3 | `unexpected disconnect`，退出 124 |
| CMA `--filter=blob:none` fetch | **成功**（tree 可用） |
| CMA `git checkout -f FETCH_HEAD` ×3 | 超时（惰性逐 blob 下载，79 个文件含大媒体） |
| CMA 按 blob SHA 逐个 `cat-file` | **成功**，38/38，耗时约 80 s |
| Recuris 完整 clone | 成功（4 MB，链路无压力） |
| MemHarness 首次 clone | 中断（仅 `.git` 12 MB）→ `--depth 1` 重试成功 |

经验：这个环境下**单个 bash 调用实际有效窗口约 175 s**（配置的上限更高但会被截断），所以任何超过 3 分钟的网络操作必须设计成可续传（`--depth 1` / `--filter` / 逐文件）。

---

## 5. 对 EvoMemory 开发的约束

1. 本机（沙箱）**没有 GPU**，所有设计必须能在 CPU + 纯 Python 上跑通核心闭环；任何依赖本地大模型的部分都要做成可替换的 provider。
2. **没有 DeepSeek API Key 的读取通道**（Linux 沙箱读不到 Windows 环境变量，`05_devflow-ai/.env` 里只有数据库配置）。因此所有 LLM 调用必须先经过一个**确定性离线桩**跑通测试，真实 API 只在 key 就位后启用，且计入 200 次 / 50 万 token 预算。
3. 第三方代码的执行必须先审 import 与副作用（本次对 CMA 测试的审查即为例行流程）。
4. 网络操作必须可续传，单步控制在 3 分钟内。
