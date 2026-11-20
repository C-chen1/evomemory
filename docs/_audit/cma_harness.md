# CMA-Harness 源码审计

- 仓库：`caseclose/cma-harness`（GitHub），论文 arXiv 2607.08497《Cognitive-structured Multimodal Agent for Multimodal Understanding, Generation, and Editing》，作者 Feng Wang（北大）/ Canmiao Fu、Zhipeng Huang、Chen Li、Jing LYU（腾讯微信视觉）/ Ge Li（北大）。
- 审计对象：`D:\python\PyCharm\Project\06_evomemory\references\cma-harness-files\` 下 38 个抽取文本文件；完整文件清单用 `git ls-tree -r --name-only FETCH_HEAD` 只读获得（79 个路径，其中 `research/` 38 个）。
- 仓库只有一个 commit：`84732bc Release portable CMA training, inference, and evaluation research code`（`git log`），没有论文代码的历史迭代。
- 标注约定：**【读到】**=代码/文档原文可指认；**【推测】**=基于代码结构但无直接证据；**【计算】**=按公开模型规格估算，代码中并无该数字。

## 0. 一句话结论

这是一套"**把视觉历史压缩成文本记忆 + 让 VLM 自己读全部缩略图和文本挑 rank**"的系统：记忆里没有 embedding、没有向量索引、没有遗忘/合并，检索是每轮对**全量记忆**做一次线性扫描；"压缩"体现在 448 缩略图、tags、description 三种可选表示上，但代码里**没有任何 token 成本量化**。许可证上**没有给任何授权**，代码不能搬。

## 1. 核心问题

**【读到】** 根 README:68 把系统定位为 "memory-centric multimodal agent that externalizes visual history into **Episodic Visual Memory (EVM)**, selectively retrieves relevant visual episodes, and plans ... through a **Multimodal Executive Controller (MEC)**"。

长程多模态记忆的具体痛点，从代码可以反推为三条：

1. **上下文窗口装不下**：`run_kv_cache_baseline.py:1-8` 的基线把"第 1~n-1 轮的全部对话 + 全部图片"直接喂给模型；20 轮会话下每轮重放历史图，视觉 token 随轮数线性增长。
2. **要检索的是"视觉证据"而不是文本**：用户说"那张有月亮的图"，模型必须在历史图里定位到具体某一轮的那一张，输出的是索引（rank），不是文本片段。
3. **任务类型混合**：同一会话里 generate / edit / composite / understand / pure_chat 交替（`multimodal_agent_service.py:529` 的五分类 prompt），检索策略必须随任务类型变化（生成任务可以完全不检索，编辑任务必须检索参考图）。

**【读到】** README:153-168 定义 M2CA-Bench：100 会话 × 20 轮 = 2000 轮，8 领域 × 55 话题，4 档难度（easy / medium / hard / very_hard），并专门设计 "high-similarity confounders" 与 "negative retrieval samples" 来阻断 shortcut learning。

## 2. 系统架构（代码还原）

三个 VLM 分工，加两个图像模型：

| 角色 | 名称 | 职责 | 证据 |
| :-- | :-- | :-- | :-- |
| VLM A | Memory Writer / PAE | 看新图，写记忆（tags+desc 或 title+desc+diff） | `multimodal_agent_service.py:415-457`；`run_kv_cache_trained_0204_thumbs_448_hard.py:83-99` |
| VLM B | CoRE Retriever | 读记忆库 + 用户 query，输出相关 rank 列表 | `multimodal_agent_service.py:579-640` |
| VLM C | MEC Controller | 把用户指令改写成生成/编辑 prompt，或回答 MMU | `multimodal_agent_service.py:501-507, 744-805` |
| 执行器 | Qwen3-VL（8B/32B/235B 可选） | 理解类任务的最终回答 | `run_kv_cache_baseline.py:117`；`multimodal_agent_service.py:248` |
| 生成器 | Qwen-Image / Qwen-Image-Edit-2511 + Lightning LoRA（4 步） | 文生图 / 图生图 | `run_kv_cache_baseline.py:241-320, 423-459`；`multimodal_agent_service.py:326-356` |

一次 turn 的闭环（`multimodal_agent_service.py:744-840`）：

```
用户指令
  → (可选) VLM B 检索 memory_bank，得到 rank 列表
  → 有检索结果：VLM C 走 edit prompt 重写；无：走 generate prompt 重写
  → Qwen-Image(-Edit) 生成图
  → VLM A 对**新生成的图**做摘要，写入 memory_bank（追加）
  → 返回 {image_path, retrieved_image_ids, memory}
```

**【读到】** 注意：**每张生成图都会被立刻写回记忆**（`multimodal_agent_service.py:834`；`run_kv_cache_baseline_20_thumbs_448_tags_desc.py:284-287`），包括编辑产物——所以记忆库天然膨胀，且编辑图与它的参考图之间只靠 `diff` 文本暗示关系，没有引用 ID。

## 3. 关键算法还原

### 3.1 `run_kv_cache_baseline.py`（41KB，名字有误导性）

**【读到】** 它**不是** KV-cache 压缩实现，而是"全上下文 + 单模型自选"的对照基线：

- 维护 `conversation_history`（**纯文本**，assistant 侧只写 `"Generated image: 07.png"`，见 :552）与 `all_generated_images`（原图路径列表）。
- understand/chat：让 Qwen3VL 先看历史文本 + **全部历史图**，自己输出要选哪些索引（`call_qwen3vl_for_understand_selection`，:393-421），再带着选中的图回答。
- generate/edit：同样先选索引（`call_qwen3vl_for_image_selection`，:360-391），再把**选中的原图**喂给 Qwen-Image。
- 默认模型 `Qwen3-VL-235B-A22B-Instruct`（:117），默认占 7 张 GPU（:142）。

**【读到】** 一个工程缺陷：`align_image_to_model()`（:52-64）会把图片**原地 resize 成 896×896 并覆盖写回**（`img.save(image_path, quality=95)`）。这等于篡改证据文件。thumbs 变体通过 monkeypatch 绕开（`run_kv_cache_baseline_20_thumbs_448_tags_desc.py:326`）。

### 3.2 `..._20_thumbs_448_tags_desc.py`（"记忆压缩"主实验）

**【读到】** `MemoryBank`（:46-75）就是两个平行 list：`_memories[i]` 与 `_rank_to_image[i]`，`add()` 返回 rank。记忆条目字段：`{id, rank, tags[], description, image_path}`（:286）。

检索流程（`call_qwen3vl_for_retrieval_with_tags_desc`，:116-195）：

1. 把全部记忆渲染成文本行 `memory[rank]: tags: ... description: ...`；
2. 再对每条记忆的**原图**生成 448×448 缩略图（`align_image_to_model_thumb`，:27-44，缓存到 `/tmp/kv_cache_thumbs_448`）；
3. 把所有缩略图 + 文本一起喂 Qwen3VL-8B，让它输出 rank 列表。

**【读到】** 四个消融模式（:109）：`tags_desc`（默认）/ `tags_only` / `desc_only` / `thumbs_only`。注意 `thumbs_only` 仍会列出 `memory[rank]:` 行（:145-146, :165），只是不给 tag/desc 文本。

**【读到】** 关键细节：检索用缩略图，**生成用原图**（:271-277 `input_images_for_generation.append(img_path)` 是完整路径）。所以"压缩"只压缩了**检索决策**的输入，没有压缩最终喂给生成模型的证据。

### 3.3 `run_kv_cache_trained_0204_thumbs_448_hard.py`（训练后的版本）

**【读到】** 只替换 A/B 两个组件，其余流程复用 baseline：

- VLM A prompt（:88）要求输出 `{title, description, diff}`，`diff` 仅在编辑图时描述"新图相对参考图的特点"。
- VLM B（:117-152）**只吃文本**：`self.model.infer(..., images=None, ...)`（:142），把 `记忆{id}: {title} - {description} | diff: {diff}` 拼成 prompt，输出 rank。
- 这一版**完全不再使用 448 缩略图**，尽管文件名和 `base.align_image_to_model` 的替换还在（:282）。

**【推测】** 仓库里存在两套记忆 schema 并行：
- `tags + description`（英文，服务与基线在用，`multimodal_agent_service.py:424`）；
- `title + description + diff`（中文，0204 训练版在用，`run_kv_cache_trained_0204...:88`）。
这说明代码快照来自不同迭代，README 只描述了统一后的说法，源码没有对齐。

### 3.4 `multimodal_agent_service.py`（55KB，真正的"推理框架主体"）

**【读到】** 这是唯一一处把 A/B/C 串起来部署的文件（960 行），`agent_api_server.py`（270 行）用 FastAPI 暴露 `/sessions`、`/images/upload`、`/memory-bank`、`/images/generate`、`/health`、`/metrics` 六个接口（`agent_api_server.py:90-257`）。

检索（:579-640）与 3.2 的 baseline 同构，但多了两个工程约束：

- `max_desc_chars = 300`（:591, :605-606），**描述超过 300 字符截断成 "..."**；
- 缩略图按需生成（:613-615），路径存在才加，缺失就少一张图（会导致 rank 与 images 顺序错位，**代码未做对齐保护**）。

**【读到】** 生成后立刻 `summarize_image()` 写记忆（:834），`max_new_tokens=512`（:430）。

### 3.5 `vlmb_retrieval_service.py`（VLM A 的 RL 奖励服务）

**【读到】** 纯 stdlib 的 HTTP 服务（:9-16），`POST /retrieve`，一次可批处理多条请求（:113-147），单卡驻留 VLM B，供训练进程远程调用。prompt 模板与 benchmark 对齐（:54-75）。

### 3.6 奖励函数

**VLM B（检索器）RL**：`retrieval_reward_300.py:47-71`
- 完全一致 = 1.0，否则 Jaccard；
- `hard`/`very_hard` 再 ×1.2（:67-68）——**奖励可超过 1.0**；
- 空集与空集也算 1.0（`_jaccard`，:39-45）。

**VLM A（记忆写入器）RL**：`vlm_a_downstream_retrieval_reward.py:153-306`
- 把 VLM A 生成的 tags/desc **替换回候选记忆库**，再调用**冻结的** VLM B 服务，用下游检索命中率当奖励（:198-259）；
- `single_regen` 模式只替换一个 rank，`all_ranks` 模式要求一次输出全部 rank 并乘以覆盖率（:301-302）；
- 本质是"**用检索器的可检索性来监督记忆的写法**"——这是本仓库最值得借鉴的训练设计。

### 3.7 负例构造（对"负迁移"最关键）

**【读到】** 负例不是"近似图干扰"，而是**结构性负样本**：

`generate_v2_with_negatives.py:57-77` 的 user simulator 被强制要求：
> "如果你被分配的任务是 'generate'（开启新主题），你必须尝试在指令中提到历史图片（如："那天的那张图不错"），但你必须在指令的后半部分明确表达你要"抛弃之前的参考，画一个完全不相关的新东西"。这种情况下，你的 ref_image_ids 必须设为 []。"

同时规定**禁止说 "ID"/数字编号**，只能用"那个有月亮的图""最后那张"这类视觉/位置指代（:68）。

`generate_training_data.py` 的 20 轮计划（:35-78）进一步把负例、跨话题、长期记忆、多图对比写进 plan：
- `negative: True` → 约束"必须是全新场景，不能引用任何历史图片"（:188-189），且 `desired_k = 0`（:231-232）；
- `cross_topic` → 回到别的主题的图（:192-193, :281-282）；
- `long_term` → 取该主题**最早**那张图（:283-286）；
- `compare` → 强制两张不同话题的图（:259-270）。

**【读到】** 真正的"高相似度混淆"来自机制而非显式构造：同一 topic 的 `generate → edit → edit` 会产生同场景近重复图（topic pool 见 `generate_training_data.py:33`，60 个场景），记忆库里自然存在近邻。

## 4. 多模态记忆的数据模型

**【读到】** 三种 schema，字段并集如下：

```text
MemoryItem = {
  id: int            # 会话内自增，等价于 rank
  rank: int          # 仅在 benchmark 脚本里显式存
  created_at: str    # 仅 service 版有（multimodal_agent_service.py:63）
  tags: [str]        # 至少 5 个（:424 prompt 约束）
  title: str         # 仅 title 版
  description: str   # 至少 100 词（prompt 约束）
  diff: str          # 仅编辑图非空，描述"新图相对参考图的特点"
  image_path: str    # 原图绝对路径
}
```

- **原图**：始终以文件路径保留在磁盘（`/tmp/agent_generated/...` 或 benchmark 输出目录），记忆条目只存路径，不存像素。
- **缩略图**：**不是记忆字段**，是检索时按需生成、缓存在 `/tmp/agent_thumbs_448`（`multimodal_agent_service.py:25-43`），缓存键是 `md5(image_path)[:12]`——注意这里**没有**带上 mtime/size，文件被覆盖会命中旧缓存（对比 benchmark 版 `_thumb_path` 用了 mtime+size，:22-25）。
- **embedding**：**不存在**。全仓库 grep `embedding|faiss|chroma|vector|cosine` 无命中。
- **血缘与来源**：`diff` 描述了变化，但**没有 `parent_id` / `reference_image_id`**；也没有 `source`（user_upload vs generated）、`turn`、`topic`、置信度。编辑链只能靠自然语言猜。

**【读到】** 记忆库是**会话级**的：`create_session()` 新建空 `MemoryBank`（:407），`clear_memory_bank()` 清空（:515-520）。跨会话长期记忆在代码里**不存在**——README 说的 "multi-session deployment"（README:68）指的是 harness 的多会话外壳，不是持久记忆。

## 5. 记忆压缩策略：三种表示各压缩了什么

| 表示 | 压缩了什么 | 丢掉了什么 | 代码中的成本控制 |
| :-- | :-- | :-- | :-- |
| **thumbs 448** | 像素量：任意分辨率 → 448×448 | 长宽比（**直接 resize 成正方形，会拉伸变形**，:38-40）、细粒度纹理、小字/OCR、小物体 | 固定 448，无自适应、无分辨率实验 |
| **tags** | 整张图 → ≥5 个自由词 | 空间关系、数量、否定、罕见实体；无受控词表，无置信度 | 无 |
| **description** | 整张图 → "≥100 words" 的英文段落 | 精确细节、可核对性；模型幻觉无法追溯 | 检索时截断到 **300 字符**（service:591）；构数据集时 600 字符（`build_retrieval_trainset_from_benchmark.py:273`） |
| **diff** | 编辑前后 → 一句差异 | 未变的背景、参考图身份 | 无 |

**【读到】** 代码里**没有任何 token 计数、显存、延迟的量化**。全仓库只有 `max_tokens` / `max_new_tokens` 这类生成长度上限，以及字符截断。README:147 声称 "reducing visual-token overhead"，但**没有给出测量代码**。

**【计算】** 若按 Qwen-VL 系 16×16 patch + 2×2 merge 估算，448×448 单图约 196 个 visual token；20 轮记忆库满时，每轮检索输入约 196×N 个图像 token + 数百字文本。N=20 时约 3900 visual token/轮，且**随轮数线性增长**。此数字代码中不存在，仅供 EvoMemory 做预算设计参考。

**【读到】** 明显的**训练-推理不一致**：VLM A 被要求写"至少 100 词"的描述，但检索端只给 300 字符——英文 100 词约 600 字符，**一半以上永远读不到**。这是可以立即拿来做实验的缺口。

## 6. 检索与评测

### 6.1 指标

**【读到】** `calculate_retrieval_accuracy_tags_desc.py:50-59`：

```python
retrieval_target = scenario_turn['meta']['retrieval_target']
if retrieval_target is None: continue          # 无标注的轮次跳过
expected  = sorted(normalize_indices(retrieval_target) or [])
predicted = sorted(normalize_indices(turn['retrieved_ranks']) or [])
correct  += (expected == predicted)
```

- **集合完全相等**才算对，不是 Recall@k / mAP；
- `null` 与 `[]` 等价，都表示"不检索"（:25, :118）；
- 空预测也计入分母（不单列 missing，:30 注释明说 `missing` 恒为 0）。

**【读到】** README:86 明确警告："The Hard subset additionally requires the `very_hard` scenario label and must not be conflated with a plain turn-range filter." 对应 `--turn-start/--turn-end` 与 difficulty 是两套筛选。

### 6.2 会话切分

**【读到】** `split_retrieval_sessions.py`：

- **按 `sample_id` 分组后整体切分**，不是按行随机切（:25-30, :72-79）——防泄漏；默认 80/10/10（:61-63），`seed=42`，先 shuffle 再按比例切（:74-76）；
- 额外产出 `*_train_hard.jsonl`（difficulty ∈ {hard, very_hard} 的子集，:64, :80-81），并落盘 `*_split_meta.json` 记录 session_ids 与行数（:93-95），可复现。

**【读到】** 训练集不是凭空造的，而是**从评测结果回灌**：`build_retrieval_trainset_from_benchmark.py` 读 scenario + 某个 baseline 的 result，重建每轮记忆库（含 thumb 路径，:156-173），产出：
- SFT：`messages=[user(prompt), assistant("Image IDs: [...]")]` + `images`（:208-216）；
- RL：只保留 **预测错误 / hard / negative** 三类（:233-250，`--rl-include-error/hard/negative`），即"错题集强化"。

### 6.3 生成质量

**【读到】** `evaluate_gemini_generation_quality_main.py:4,51` 用 LLM-as-judge 打 **0-10 整数**，分 Overall / Generate / Edit / Understand-Chat 四桶，严格解析 `Score (0-10): N`（:53-66）。README:79 报的 8.49/10 是这套口径。

**【读到】** `eval_utils.py:6-48` 是另一套旧的 1-5 分制（MMU/Edit/Generate/Chat 各有 rubric），与上面 0-10 不是一套，属于遗留代码。

## 7. 训练相关与"无 GPU 不可能复现"清单

### 7.1 训练栈

**【读到】** 全部基于 ModelScope **ms-swift 3.12.0**，LoRA（r=8, alpha=32, all-linear, `freeze_vit true`），bf16 + flash_attn + DeepSpeed ZeRO-2：

| 脚本 | 训练对象 | 关键超参 | 证据 |
| :-- | :-- | :-- | :-- |
| `train_retrieval_sft_300.sh` | VLM B 检索器 SFT | 8 卡, 3 epoch, lr 2e-5, max_len 8192, 输入 = tags+desc+448 缩略图 | :41-58, :73-102 |
| `train_retrieval_rl_dapo_300.sh` | VLM B DAPO/GRPO | 8 卡, `num_generations=8`, lr 2e-6, reward = jaccard(+hard×1.2) + soft_overlong | :44-64 |
| `train_vlm_a_rl_dapo_300_a2.sh` | VLM A 记忆写入器 RL | **7 卡训练 + 1 卡跑冻结 VLM B 服务**, `num_generations=4` | :26, :96, :131-165 |

**【读到】** VLM A RL 的启动脚本会先轮询 `/health` 等 VLM B 服务起来（:96-127），否则直接退出——说明奖励依赖一个**独立进程的在线模型**，不是纯函数。

### 7.2 无 GPU 下根本不可能复现的部分（明确清单）

| 项目 | 为什么不可能 | 证据 |
| :-- | :-- | :-- |
| 任何 benchmark 运行 | 默认 Qwen3-VL-**235B**（4-8×40GB+）；`init_qwen3vl_model` 检测到 `gpu_count == 0` 直接 `RuntimeError('No available GPU; model cannot run on CPU')` | `run_kv_cache_baseline.py:117, 135-136, 149-150` |
| Qwen-Image 生成/编辑 + 服务化推理 | diffusers pipeline 按 GPU 逐张加载、无 CPU 分支；A/B/C 与生成器硬编码 8 卡角色（A=0,B=1,C=2,Qwen-Image=3-7） | `run_kv_cache_baseline.py:304-320`；`multimodal_agent_service.py:260-265, 334-337` |
| VLM B SFT / RL、VLM A RL | 8 卡 + ZeRO-2 + flash_attn；VLM A 还要 7 卡训练 + 1 卡奖励服务，且需要已训好的 VLM B | 两个 `.sh` 的 `NPROC_PER_NODE=8`；`train_vlm_a_rl_dapo_300_a2.sh:26, 131-165` |
| 论文分数复现 | 权重、M2CA-Bench、训练数据**均未发布** | `README.md:203`；`research/README.md:9-10` |
| 数据生成（scenario engine） | 需要理解模型 + 图像生成模型各一个（Gemini API 或本地 Qwen） | `generate_training_data.py:23-24` |

**【读到】** 作者自己承认："GPU training, full inference and paper-score reproduction have not been rerun for this export."（`research/README.md:19-20`），`SOURCE_PROVENANCE.json:4-8` 也只记录了 27 个 py 语法检查、14 个 CPU 单测、`gpu_reproduction_rerun: false`。

**【读到】** 唯一能在无 GPU 下跑的：`python -m unittest discover -s research/tests`（`research_README.md:102-104`），14 个用例覆盖奖励算术、指标边界、HTTP 适配器、脚本装配（`tests/test_release.py:30-152`）。这部分是**可以立刻借鉴的测试思路**。

## 8. 可复用组件（先看许可证，见 A 节）

**【读到】** 以下设计值得"吸收思路"：

1. **三模型职责分离**：写入器 / 检索器 / 控制器各自独立可换（`multimodal_agent_service.py:283-292`，还支持 `VLM_SHARE_MODEL=1` 合并到一张卡）。EvoMemory 可以直接照这个边界设计接口。
2. **用下游检索奖励训练记忆写入器**（`vlm_a_downstream_retrieval_reward.py`）：把"记忆写得好不好"定义为"检索器能不能靠它找对图"，而不是 BLEU/ROUGE。这是可迁移的**记忆质量目标函数**。
3. **错题集回灌 + 按会话切分**：只挑错误/hard/negative 轮次做 RL 数据（`build_retrieval_trainset_from_benchmark.py:233-250`），并按 session 切分、落盘 split meta 防泄漏（`split_retrieval_sessions.py`）。
4. **结构性负样本**（说"提到旧图但明确不用"）：比随机负例更能测出"该不检索时就不检索"（`generate_v2_with_negatives.py:68`）。
5. **奖励服务独立成进程**：把在线模型包成 HTTP 服务，训练侧只做调用（`vlmb_retrieval_service.py`），解耦显存与训练框架；同时用 CPU 单测守住奖励算术与指标边界（`tests/test_release.py`），这是无 GPU 环境下唯一的正确性防线。

**【读到】** 但不建议照抄的实现：原地覆盖图片（`run_kv_cache_baseline.py:52-64`）、缩略图缓存键不含内容哈希（`multimodal_agent_service.py:31`）、rank 与 images 顺序可能错位（:613-615）。

## 9. 改造难点

1. **没有向量检索**：全量线性扫描 + VLM 读图，N=20 尚可，N=200/2000 时 prompt 会爆。要移植必须自己补"粗筛 → 精排"两阶段。
2. **记忆只增不减**：无合并、无遗忘、无去重、无更新（grep `forget|evict|merge|prune` 仅命中数据集重建时的 `_update_memory_bank`）。EvoMemory 的"演化"必须从零写。
3. **血缘缺失**：`diff` 是自然语言，没有 `parent_id`，无法做版本树、回滚、冲突检测。
4. **两套 schema 并存**（tags+desc vs title+desc+diff），迁移时得先决定统一口径。
5. **训练-推理不一致**：写入要求 100 词、读取截断 300 字符；缩略图在训练里是输入、在 0204 推理里被弃用。
6. **强 GPU 耦合 + 无数据/权重**：8 卡角色硬编码，单卡/CPU 适配要重写初始化路径；复现还得先自造 20 轮会话数据（scenario engine 依赖付费 API 或本地大模型）。

## A. 许可证与可复用性（结论：**不能搬代码**）

### A.1 事实

- **仓库根目录及全仓库没有任何 `LICENSE` / `COPYING` 文件**。`git ls-tree -r --name-only FETCH_HEAD | grep -i licen` 只命中 `research/THIRD_PARTY_NOTICES.md` 与 `research/third_party/ms-swift-LICENSE`（Apache-2.0 **原文**，201 行，是 ms-swift 的许可证副本）。
- `research/THIRD_PARTY_NOTICES.md` 全文只有 8 行，原文如下（逐字引用）：

> The training entrypoints and reward-plugin API depend on ModelScope ms-swift.
> Its Apache-2.0 license is retained in third_party/ms-swift-LICENSE. The framework
> itself is installed separately and is not vendored in this source release.
> Qwen model weights, image models, PyTorch, Diffusers, Transformers and other
> dependencies retain their own licenses and terms. No model weights are included.
> **This file does not assign a new license to the authors' research code.**

（加粗为引用时强调，原文无加粗；见 `research_THIRD_PARTY_NOTICES.md:3-8`。）

### A.2 法律含义

1. **"公开可见" ≠ "获得授权"**。著作权默认归作者所有（《伯尔尼公约》、中国《著作权法》），除非权利人明确授予许可。仓库没有 LICENSE，就是**保留所有权利（all rights reserved）**，即使它挂在 GitHub 上。
2. **那行关键声明是在"反向澄清"**：作者特地说明 `THIRD_PARTY_NOTICES.md` 这份文件**不构成**对研究代码的授权。也就是说，Apache-2.0 只覆盖 `third_party/ms-swift-LICENSE` 对应的 ms-swift 框架，**不覆盖** `research/` 下任何 `.py` / `.sh` / prompt 文本。
3. **GitHub 服务条款**只授予"在 GitHub 平台内查看与 fork"的权利，不授予复制进自己项目、修改、再分发、商用的权利。
4. **可迁移的只有"思想"**：算法、架构、数据模型、指标定义属于思想/方法，不受著作权保护（思想-表达二分法）。**代码、注释、prompt 原文、文件组织**属于表达，受保护。

### A.3 明确结论

- **不能**把 `research/` 下的代码（含 `.py`、`.sh`、prompt 字符串）复制、改写、粘贴进 EvoMemory，也不能随项目分发，无论是否注明出处。
- **可以**：阅读、学习、在论文/文档中引用（附 arXiv 编号与仓库链接）、**独立重新实现**同一思想并用自己的代码与措辞。
- **建议做法**：EvoMemory 里凡受其启发的设计，写成"参考 CMA（arXiv 2607.08497）的思路，独立实现"，并保留引用；prompt 一律自己重写，不要逐字复用。
- **如需正式授权**：README:13 给了作者邮箱 `fengwang@stu.pku.edu.cn`，可直接询问是否可加 MIT/Apache-2.0。在这之前，按"不可复用"处理。
- 说明：以上是工程判断，不构成法律意见；若涉及商用，请咨询律师。

## B. 核心代码到底发布了没有

### B.1 README 的表述

**【读到】** **没有 "code coming soon" 之类的话**。相反：

- `README.md:201`："The **CMA research code** is now available: training launchers, reward functions, inference services, scenario-engine source, and evaluation scripts."
- `README.md:203`："**M2CA-Bench, the training data, and trained model checkpoints will be made publicly available upon acceptance.**"（注意：论文已挂 arXiv，但这里仍写 "upon acceptance"，即**数据与权重尚未发布**）
- `research/README.md:14-20`："The code is an exported research snapshot. It is separate from the extended CMA-Harness deployment and **does not contain its complete 17-tool integration**. ... the private fork is not vendored. ... GPU training, full inference and paper-score reproduction **have not been rerun** for this export."

### B.2 基于文件清单与代码内容的判断

**结论：属于"部分发布"——推理/记忆/检索/控制的主干在，但支撑复现的三件套（数据、权重、私有框架）全缺，而且 harness 的 17 工具集成不在。**

已发布（可读、结构完整）：

- 记忆/检索/控制推理服务：`multimodal_agent_service.py`（960 行）+ `agent_api_server.py`（270 行）+ `qwen_lightning_generator.py`；
- 三套 benchmark runner（全上下文基线 / tags+desc 记忆 / 训练后记忆）与两套检索指标脚本；
- 两个奖励插件 + 三个训练 launcher；
- scenario engine 源码（`gemini_multi_agent/` 5 个文件，含 60 话题池与 20 轮计划）；
- 14 个 CPU 单测。

**未发布 / 缺失（关键）**：

- 模型权重（`VLM_A_PATH/VLM_B_PATH/VLM_C_PATH` 全指向不存在的 `./checkpoints/*`，`research/.gitignore:4` 已排除 `checkpoints/`）；
- M2CA-Bench 的 scenario jsonl（`research/.gitignore:6` 排除 `scenarios/`；benchmark 脚本默认目录 `scenarios/50x20-gemini-hard` 在仓库中不存在）；
- 训练数据（`research/.gitignore:5` 排除 `datasets/`）；
- 私有 ms-swift fork（只保留项目代码，需自行安装公开 3.12.0）；
- 17 工具集成（`research/README.md:15` 明说）；
- **VLM C（控制器）的训练脚本**（只有推理调用，三个 launcher 里没有 C）；
- **VLM A 的 SFT launcher**（只有 A 的 RL；虽然数据生成器产出了 A 的 SFT 数据）。

**是否可以"跑起来"**：

- 无 GPU：只能跑 14 个单测，**不能**跑任何 benchmark 或服务。
- 有 8 卡：理论上能拉起服务与 runner，但由于**没有 checkpoint 与 scenario 文件**，只能拿公开 Qwen 基座 + 自造数据跑通流程，**得不到论文任何数字**。
- 作者自述："paper-score reproduction have not been rerun for this export."（`research/README.md:19-20`）

**判断**：这不是"代码即将开源"的占位仓库，而是一次**真实的、经过语法与单测检查的研究快照发布**；但它发布的是"**方法可读性**"而非"**结果可复现性**"。对 EvoMemory 的价值在"看清它怎么组织记忆与检索"，不在"跑它的数字"。

## 10. 对 EvoMemory 的启示

以下只吸收思路，不吸收代码（理由见 A 节）。

### 10.1 可吸收的 6 条设计

1. **记忆写入与检索解耦成两个可训练角色**，并用"下游检索命中率"作为写入器的奖励（对应 `vlm_a_downstream_retrieval_reward.py` 的思想）。EvoMemory 可以把它推广成"经验条目写得好不好 = 未来任务能不能用它"。
2. **区分"检索表示"与"执行证据"**：CMA 用 448 缩略图做检索、用原图做生成（`run_kv_cache_baseline_20_thumbs_448_tags_desc.py:271-277`）。EvoMemory 可以把它升级成**分层证据**（缩略图 → 局部裁剪 → 原图），并按任务预算逐级取用，而不是二值选择。
3. **结构性负样本**：让用户"提到旧图但明确不用"，专门训练"该不检索时就不检索"。这比随机负例更能防"过度检索"。
4. **错题集回灌**：只把预测错误 / hard / negative 的轮次放进 RL 数据（`build_retrieval_trainset_from_benchmark.py:233-250`），样本效率高且天然聚焦难例。
5. **按会话切分 + 落盘 split meta**：EvoMemory 的 A/B/C/D 实验分组应默认沿用"按会话切、禁跨切分泄漏"（`split_retrieval_sessions.py`）。
6. **奖励服务独立成进程**：把在线模型包成 HTTP 服务，训练侧只做调用（`vlmb_retrieval_service.py`），解耦显存与训练框架；同时用 CPU 单测守住奖励算术与指标边界（`tests/test_release.py`），这是无 GPU 环境下唯一的正确性防线。

### 10.2 "多模态记忆压缩"上它没做、我能做的实验

1. **量化 token-精度曲线（它完全没做）**：CMA 只给了 4 个定性消融模式（tags_desc / tags_only / desc_only / thumbs_only），**从未统计 token 数、延迟、显存与准确率的取舍**。EvoMemory 可以做"压缩率 vs 检索准确率"的帕累托曲线：每张图 196 / 784 / 3136 visual token 三档 × 文本 0/50/150/300 字符，画出成本-收益前沿。
2. **缩略图分辨率与长宽比**：代码固定 448 且**直接拉成正方形**（`_thumb_path` / `align_image_to_model_thumb`，会变形），无任何消融。可做 224/448/896 × {正方形拉伸, letterbox, 中心裁剪} 的 2×3 网格，看"变形损失"到底值多少准确率。
3. **训练-推理截断错配**：写入端要求 ≥100 词，读取端截 300 字符（service:591），**一半信息永远读不到**。可做"写入长度 × 读取预算"二维实验，直接量化"写了不读"的浪费。
4. **记忆库规模的退化曲线**：CMA 每会话最多约 20 张图，从未测 N=50/100/500。可测"线性扫描 VLM 检索"的准确率与延迟随 N 的衰减，作为引入粗筛（文本/向量）的动机实验。
5. **压缩表示对"编辑一致性"的影响**：CMA 的压缩只作用于检索，生成仍用原图，因此**没有回答"如果连生成也只能看压缩表示会怎样"**。EvoMemory 可以做"证据降级"实验：检索与生成都用 thumb / 都用 desc / 混合，测编辑保真度下降曲线——这正是"记忆压缩"真正的代价所在。
6. **负迁移的显式测量（它只有结构性负例）**：CMA 的负例是"不该检索"，不是"检索到了错的记忆"。可注入"看似相关但实为旧版本"的记忆（利用缺失的血缘关系），测**错误记忆被采纳率**与任务失败率，这是 EvoMemory 相对 CMA 最有区分度的实验。
