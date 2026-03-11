# Pgramma

[English](README.md)

Pgramma 是一个本地优先的数字人格引擎：单个 `.pgram` 文件保存人格身份、记忆与对话历史。

## 核心概念

- 一个文件，一个人格。
- 记忆只追加，不回写。
- 每轮流程：召回 -> 回应 -> 评估 -> 存储。

## 快速开始

### 前置要求

- Rust 1.80+（edition 2024）
- 至少一个可用的 LLM Provider 接口

### 安装与运行

```bash
git clone https://github.com/KercyDing/pgramma.git
cd pgramma
cp config.example.toml config.toml
# 编辑 config.toml：设置 active_provider、model、api_key
cargo run
```

## 命令

| 命令 | 说明 |
|------|------|
| `/help` | 查看命令 |
| `/test <path>` | 从文件导入并验证召回 |
| `/stats` | 查看 engram / episode 统计 |
| `/quit` | 退出 |

## 配置说明

以 `config.example.toml` 作为模板，本地复制为 `config.toml` 后修改。

关键点：

- `llm.active_provider` 选择当前 Provider。
- 当前 Provider 的 `llm.providers.<provider>.model` 与 `api_key` 必填。
- `embedding.model_id` 控制本地 embedding 模型。
- `config.toml` 已加入 git ignore。

缺少必填字段时，程序会输出 fatal 信息并优雅退出。

## 架构概览

- `src/main.rs`：REPL 与启动流程
- `src/config.rs`：TOML 配置解析
- `src/db/`：基于 redb 的存储
- `src/llm/`：多 Provider 推理与评估
- `src/memory/`：向量与召回流程

## 测试

```bash
cargo test
```

## 许可证

[GPL-3.0](LICENSE)
