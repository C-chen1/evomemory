# SECURITY_AUDIT

审计时间：2026-11-21（第二阶段 P0）
审计范围：工作区、Git 历史、artifacts / logs / data 目录、记忆目录。

---

## 1. 结论

| 检查项 | 结果 | 证据 |
|---|---|---|
| 当前工作区被跟踪文件是否含密钥 | **干净** | `git grep` 全历史扫描无命中；`tests/test_secrets.py` 全绿 |
| Git 历史是否曾提交过密钥 | **从未** | `git log --all -p -- .env.example .env \| grep '^+DEEPSEEK_API_KEY=.\+'` 无输出 |
| `.env` 是否被忽略 | 是 | `git check-ignore -v .env` → `.gitignore:2:.env` |
| `.env` 是否被跟踪 | 否 | `git ls-files | grep -c '^\.env$'` → 0 |
| `.env.example` 的值是否为空 | 是 | `tests/test_secrets.py::test_env_example_has_empty_values` |
| artifacts / logs / data 是否含密钥 | **干净** | `grep -rlE 'sk-[A-Za-z0-9]{20,}' artifacts logs data` 无命中 |
| 长期记忆目录是否含密钥 | **干净** | 同模式扫描 `D:\AI\Claude Code\Devflow AI长期记忆` 无命中 |

**Git 历史未被污染，不需要重写历史。** 这一点是实测结论，不是推断：写进 `.env.example` 的那次发生在 commit `a212f2f` 之后、`0e728af` 之前，而 `a212f2f` 里该行的值是空的。

---

## 2. 曾经发生的两次事故（记录以便复发检测）

### 事故 1：密钥被写进 `.env.example`

`.env.example` 是**被 Git 跟踪**的模板文件。用户把真实密钥填入了它，而不是 `.env`。
发现时它尚未被提交（`git status` 显示为未暂存修改），因此没有进入历史。

处置：密钥移入 `.env`（已被忽略），模板恢复为空值。

### 事故 2：密钥出现在会话记录中

密钥曾在会话里以明文形式出现（系统提示中展示文件改动内容时触发）。
这不在代码仓库的风险面内，但意味着**密钥应按已泄露处理**。

---

## 3. 需要人工完成的事项

| 事项 | 原因 | 我不能做的事 |
|---|---|---|
| **轮换第一个密钥**（曾写入 `.env.example`，36 字符纯 UUID） | 已出现在会话记录中 | 无权操作 API 后台 |
| **轮换第二个密钥**（当前在用的 `sk-` 密钥） | 同样出现在会话记录中 | 同上 |
| 轮换后更新 `.env` | 新密钥只能由用户填写 | 我不会写入或打印密钥 |

两个密钥都应视为已泄露。当前 `.env` 里的那个仍可用，但它的保密性已经不能假设。

---

## 4. 自动化防线（已落地）

`tests/test_secrets.py` 的四个测试在每次 `pytest` 时执行：

1. `test_no_secret_shaped_strings_in_project_files` — 扫描全部项目文件（排除 `.git` / `references` / `artifacts` / `data` / 缓存目录），匹配 `sk-…`、`api_key=…`、UUID 三种形状。
2. `test_env_example_has_empty_values` — 断言 `.env.example` 的 `DEEPSEEK_API_KEY` / `DEEPSEEK_MODEL` 必须为空。
3. `test_env_is_gitignored` — 断言 `.env` 在忽略列表且 `.env.example` 被显式排除在外。

局限（如实记录）：这些测试只覆盖**工作区**。它防不住 `git commit -a` 之前手动 `git add -f`，也防不住密钥出现在 `artifacts/`（被忽略但仍是明文）。因此轮换才是根本手段，测试只是降低复发概率。

---

## 5. Agent 侧的最小权限

- `.env` 只有 `Settings.from_env()` 读，`describe()` 只输出 `api_key_present` 与末四位指纹。
- `llm_calls` 表记录 model / purpose / tokens / latency / error，**不记录 prompt 内容与密钥**。
- `scripts/vision_smoke.py` 不对图片内容做任何持久化输出，只保存生成图与判分结果。
- benchmark 的 answer 文件不出现在 agent 可见的工具里（工具只有 `list_records` / `get_record` / `set_field` / `log_entry`）。
