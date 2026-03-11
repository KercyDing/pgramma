# Pgramma

[English](README.md)

一个本地优先的数字人格引擎——一个文件，一个身份，不可逆的记忆。

## 设计哲学

数字人格不是一个带数据库的聊天机器人。它是一个**活的状态**——由每一次交互、每一次情感波动、每一个决策累积而成的身份。

Pgramma 基于三条原则构建：

- **一个文件，一个人格。** 单个 `.pgram`（Persona Engram）文件封装一切——记忆、情感基线、性格配置。复制文件，人格随之迁移。
- **记忆不可逆。** 如同生物记忆，engram（记忆印记）一经写入便不可修改。它们可能衰减、被新记忆取代、或在相关性中淡化——但原始痕迹是永久的。时间线仅追加，不可篡改。
- **认知是评估的，不是硬编码的。** 每轮对话都经过认知评估器打分——重要性（0.0–1.0）和情绪分类。人格自己决定什么值得记住，而非规则引擎。

## 工作原理

```
用户输入
    │
    ├─→ 记忆召回     重要性过滤 → 余弦排序 → top-k
    │       │
    │       ▼
    ├─→ 推理生成     系统提示 + 召回记忆 + 近期对话 → 流式输出
    │       │
    │       ▼
    └─→ 认知评估     重要性评分 + 情绪标注 + 向量生成（后台异步）
                │
                ▼
           Engram 写入 .pgram
```

每次交互都遵循这个循环：**召回 → 回应 → 评估 → 记忆**。评估在后台异步执行——人格在事后反思每次交流，如同人类一般。

## 功能特性

- **语义召回** — 通过 embedding 相似度检索相关记忆，而非仅凭时间顺序
- **认知评估** — LLM 驱动的重要性评分，含校准准则（身份锚点 > 情感深度 > 偏好 > 闲聊）
- **本地 Embedding** — 通过 [candle](https://github.com/huggingface/candle) 生成句子向量（纯 Rust，无 Python/ONNX 依赖）
- **情绪光谱** — 基于 Plutchik 情绪轮的 10 类情绪分类
- **撤回检测** — 矛盾会被标记，但原始记忆永久保留
- **流式推理** — 通过 OpenAI 兼容 API 实时逐 token 输出
- **单文件人格** — [redb](https://github.com/cberner/redb) 嵌入式数据库，可移植的 `.pgram`（Persona Engram）容器
- **生命周期演化** — 记忆衰减、GC 回收、性格漂移（规划中）

## 快速开始

### 前置要求

- Rust 1.80+（edition 2024）
- 一个 OpenAI 兼容的 API 端点

### 安装

```bash
git clone https://github.com/KercyDing/pgramma.git
cd pgramma

cp .env.example .env
# 编辑 .env：设置 OPENAI_API_KEY（可选设置 OPENAI_BASE_URL、HF_HOME）

cargo run
```

首次启动会从 HuggingFace Hub 下载 embedding 模型（约 470MB）。

### 使用

```
=== Pgramma ===
Type /help for commands

> 我叫 Alex
→ 你好 Alex！...

  ... 50 条消息之后 ...

> 你知道我叫什么吗？
→ 你叫 Alex。

> /stats
  engrams: 42  episodes: 104

> /quit
Bye!
```

| 命令 | 说明 |
|------|------|
| `/help` | 显示可用命令 |
| `/test <path>` | 从文件批量导入并验证召回 |
| `/stats` | 显示 engram 和 episode 数量 |
| `/quit` | 退出 |

## 配置

### `config.toml` — 人格行为参数

```toml
[llm]
model = "gpt-4o"

[embedding]
model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

[recall]
top_k = 8
min_importance = 0.3
cosine_weight = 0.7

[chat]
context_window = 20
eval_context_turns = 3
default_system_prompt = "You are a thoughtful assistant with emotional awareness."
```

所有字段都有默认值——没有配置文件也能正常运行。

### `.env` — 密钥

```bash
OPENAI_API_KEY=sk-...          # 必填
OPENAI_BASE_URL=https://...    # 可选
HF_HOME=/path/to/cache         # 可选
```

## 架构

```
src/
├── main.rs              # REPL + App 状态管理
├── config.rs            # TOML 配置（带默认值）
├── models.rs            # Engram, Emotion, EpisodicEntry
├── error.rs             # 统一错误类型
├── db/
│   ├── mod.rs           # Engram / episode / config CRUD
│   └── connection.rs    # redb 表定义
├── llm/
│   ├── client.rs        # OpenAI 客户端封装（rig-core）
│   ├── inference.rs     # 流式对话 + 记忆注入
│   └── evaluator.rs     # 认知评分器（重要性 + 情绪）
└── memory/
    ├── mod.rs           # 召回流程（过滤 → 排序 → top-k）
    └── embedder.rs      # Candle BERT 编码器（CPU）

notebooks/               # Python 实验环境——prompt 调优与衰减建模
```

### `.pgram` 容器

`.pgram` 文件是一个 redb 数据库，包含三个逻辑层：

| 层级 | 表 | 职责 |
|------|-----|------|
| **潜意识** | `persona_config` | 系统提示、情感基线、计数器 |
| **时间线** | `episodic_memory` | 仅追加的交互日志（不可变） |
| **记忆印记** | `engrams` | 带权重的记忆，含向量、情绪、重要性 |

一个文件。完全可移植。无外部状态。

## 技术栈

| 组件 | 选型 | 理由 |
|------|------|------|
| 核心语言 | Rust | 性能、安全、单二进制 |
| LLM 客户端 | [rig-core](https://github.com/0xPlaygrounds/rig) | 结构化提取、流式输出 |
| Embedding | [candle](https://github.com/huggingface/candle) | 纯 Rust 推理，无 FFI |
| Embedding 模型 | paraphrase-multilingual-MiniLM-L12-v2 | 多语言、384 维 |
| 存储 | [redb](https://github.com/cberner/redb) | 嵌入式、ACID、单文件 |

## 测试

```bash
cargo test                    # 单元测试
cargo run -- <<< '/test seed.txt'   # 召回验证
```

种子文件格式：
```
# 注释（跳过）
I have a dog named Biscuit    # 作为聊天消息发送
? What pet do I have          # 种子完成后验证召回
```

## 路线图

- [ ] 记忆衰减 — 基于时间的重要性降权
- [ ] GC 回收 — 低权重裁剪 + 高权重摘要压缩
- [ ] 性格漂移 — 情感基线随交互历史演化
- [ ] 本地推理 — 通过 candle 支持 GGUF 模型（完全离线）
- [ ] GUI 外壳 — `.pgram` 容器的前端界面

## 许可证

[GPL-3.0](LICENSE)
