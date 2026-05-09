# 灵智编排引擎 - LLM Agent DAG 编排系统

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

基于 LangChain 和 LangGraph 构建的企业级 LLM Agent 编排平台，采用分层、并行、动态（HPD）设计理念，实现计算资源按需分配和复杂任务高效解耦。

---

## 一、产品概述

### 1.1 产品名称
**灵智编排引擎** (HPD-Agent - Hierarchical Parallel Dynamic Agent) - LLM Agent DAG 编排系统

### 1.2 产品定位
基于 LangChain 和 LangGraph 构建的企业级 LLM Agent 编排平台，采用分层、并行、动态（HPD）设计理念，实现计算资源按需分配和复杂任务高效解耦。

### 1.3 核心价值主张
| 维度 | 价值 |
|------|------|
| **分层路由** | 根据任务难度智能路由到不同层级的处理节点 |
| **并行调度** | 基于 DAG 的任务并行执行，最大化资源利用率 |
| **动态执行** | 专家模式下的多路径生成和多维评分优化输出质量 |
| **多模态支持** | 支持文本、图片、音频、文件等多种输入类型 |
| **RAG增强** | 支持检索增强生成，实现知识问答和文档理解 |

---

## 二、系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                       HPD-Agent 架构                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐                                              │
│  │    CLI/API   │  外部入口                                    │
│  └──────┬───────┘                                              │
│         │                                                      │
│         ▼                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Agent主入口│◄───│  依赖注入    │◄───│   配置管理   │      │
│  │   (agent.py)│    │   (DI容器)   │    │  (config.py) │      │
│  └──────┬───────┘    └──────────────┘    └──────────────┘      │
│         │                                                      │
│         ▼                                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   核心模块层                            │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  │    │
│  │  │  Router │→│Scheduler│→│  Expert │→│ Multimodal│  │    │
│  │  │ (路由)  │  │ (调度)  │  │ (执行)  │  │  (处理)  │  │    │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│         │                                                      │
│         ▼                                                      │
│  ┌──────────────┐    ┌──────────────┐                          │
│  │    Cache     │    │    LLM工厂   │                          │
│  │ (缓存层)     │    │  (多厂商)    │                          │
│  └──────────────┘    └──────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 模块职责划分

| 模块 | 职责 | 文件路径 |
|------|------|----------|
| **Agent** | 核心编排框架，集成所有模块 | `hpd_agent/agent.py` |
| **路由模块** | 分层路由机制，任务难度分析 | `hpd_agent/routing/router.py` |
| **DAG调度器** | 任务并行调度，拓扑排序 | `hpd_agent/dag/scheduler.py` |
| **专家执行器** | 多路径生成，多维评分 | `hpd_agent/expert/executor.py` |
| **多模态处理** | 多类型输入处理 | `hpd_agent/multimodal/processor.py` |
| **LLM工厂** | 多厂商LLM适配 | `hpd_agent/llm/factory.py` |
| **缓存模块** | 结果缓存机制 | `hpd_agent/cache/` |
| **依赖注入** | IoC容器管理 | `hpd_agent/di/` |

---

## 三、核心功能模块

### 3.1 分层路由机制 (Router)

**功能描述**：根据任务难度自动路由到不同层级的处理节点

**路由策略**：
| 难度区间 | 层级 | 节点类型 | 适用模型 |
|----------|------|----------|----------|
| [0, 0.3) | Level 1 | Simple Node | gpt-3.5-turbo |
| [0.3, 0.7) | Level 2 | Complex Node | gpt-4 |
| [0.7, 1.0] | Level 3 | Expert Node | gpt-4-turbo |

**难度分析算法**：
- 文本长度评分 (30%)
- 关键词匹配评分 (40%) - 包含 analyze, explain, design, develop 等专业词汇
- 句子数量评分 (20%)
- 上下文历史评分 (10%)

**文件位置**：`hpd_agent/routing/router.py`

---

### 3.2 DAG并行调度引擎 (Scheduler)

**功能描述**：基于 Kahn 算法的拓扑排序，实现任务的并行调度和依赖管理，支持超时控制、重试机制和优先级调度。

**DAG架构设计**：

```
任务A ──────┐
            ▼
任务B ────→ 任务C ────→ 任务D
     │            │
     └────────────┘
```

**核心特性**：
| 特性 | 说明 |
|------|------|
| **拓扑排序** | Kahn算法保证任务按依赖顺序执行，检测循环依赖 |
| **并行执行** | 同一层级任务可并行执行，最大化资源利用率 |
| **超时控制** | 可配置单个任务超时时间，避免长时间阻塞 |
| **重试机制** | 失败任务自动重试，可配置重试次数和延迟 |
| **优先级调度** | 支持任务优先级排序，高优先级任务优先执行 |
| **难度排序** | 同层级任务按难度降序执行，优先处理复杂任务 |

**Kahn拓扑排序算法流程**：
```
1. 计算每个节点的入度（依赖数量）
2. 将入度为0的节点加入队列
3. 依次处理队列中的节点，减少其邻接节点的入度
4. 将新入度为0的节点加入下一层级队列
5. 重复直到所有节点处理完毕或检测到循环
```

**任务状态流转**：
```
PENDING → RUNNING → COMPLETED
                  ↘
                   → FAILED (可重试) → RUNNING
```

**任务模型（Task）**：
| 字段 | 类型 | 说明 |
|------|------|------|
| id | str | 任务唯一标识（UUID） |
| name | str | 任务名称 |
| type | TaskType | 任务类型（SIMPLE/COMPLEX/EXPERT） |
| difficulty | float | 难度评分 [0,1] |
| priority | int | 优先级 [0,10] |
| status | TaskStatus | 任务状态 |
| dependencies | List[str] | 依赖任务ID列表 |
| payload | Dict | 任务负载数据 |
| result | Any | 执行结果 |
| error | str | 错误信息 |
| retry_count | int | 已重试次数 |
| start_time | datetime | 开始时间 |
| end_time | datetime | 结束时间 |

**调度器配置参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| max_workers | int | 10 | 最大并发工作线程数 |
| default_timeout | int | 300 | 默认超时时间(秒) |
| retry_attempts | int | 3 | 最大重试次数 |
| retry_delay | float | 1.0 | 重试延迟系数 |

**调度流程**：
```python
# 1. 构建任务DAG
tasks = scheduler.build_dag_from_tasks([
    {"name": "Task A", "type": "simple", "dependencies": []},
    {"name": "Task B", "type": "complex", "dependencies": ["task_a_id"]},
    {"name": "Task C", "type": "expert", "dependencies": ["task_a_id", "task_b_id"]}
])

# 2. 拓扑排序
levels = await scheduler.kahn_topological_sort(tasks)
# Output: [["task_a_id"], ["task_b_id"], ["task_c_id"]]

# 3. 执行调度
results = await scheduler.schedule(tasks, executor)
```

**文件位置**：`hpd_agent/dag/scheduler.py`, `hpd_agent/dag/task.py`

---

### 3.3 动态专家执行模块 (Expert Executor)

**功能描述**：针对高难度任务的多路径生成和质量优化

**核心功能**：

| 功能 | 实现说明 |
|------|----------|
| **Prompt重写** | 将简单问题转化为专业分析型prompt |
| **动态权重评估** | 根据任务类型分配分析权重 |
| **多路径生成** | 并行生成多条推理路径 |
| **多维评分** | 从长度、结构、相关性等维度评分 |

**文件位置**：`hpd_agent/expert/executor.py`

---

### 3.4 多模态输入处理 (Multimodal Processor)

**功能描述**：支持多种输入类型的统一处理

**支持的输入类型**：
| 类型 | 处理方式 | 置信度 |
|------|----------|--------|
| 文本 | 直接使用 | 1.0 |
| 图片 | OCR识别/描述生成 | 0.85 |
| 音频 | 语音转文字 | 0.8 |
| 视频 | 帧提取+描述 | 0.7 |
| 文件 | 内容解析 | 0.9 |

**文件位置**：`hpd_agent/multimodal/processor.py`

---

### 3.5 LLM工厂 (LLM Factory)

**功能描述**：支持多种LLM提供商的统一接口

**支持的提供商**：
| 提供商 | 配置标识 | 模型示例 |
|--------|----------|----------|
| OpenAI | `openai` | gpt-4, gpt-3.5-turbo |
| DashScope | `dashscope` | qwen-7b, qwen-14b |
| Anthropic | `anthropic` | claude-3 |
| Google | `google` | palm-2 |
| Local | `local` | 本地部署模型 |
| DeepSeek | `deepseek` | deepseek-chat |

**文件位置**：`hpd_agent/llm/factory.py`

---

### 3.6 缓存系统 (Cache)

**功能描述**：多级缓存机制提升响应速度

**缓存策略**：
| 缓存类型 | TTL | 用途 |
|----------|-----|------|
| LLM响应缓存 | 1小时 | 避免重复调用 |
| 任务结果缓存 | 30分钟 | 复用任务结果 |

**后端支持**：
- 内存缓存 (MemoryCache)
- Redis缓存 (RedisCache)

**文件位置**：`hpd_agent/cache/`

---

### 3.7 依赖注入容器 (DI Container)

**功能描述**：IoC容器实现模块解耦

**核心特性**：
- 服务注册与解析
- 工厂模式支持
- 单例管理
- `@inject`装饰器自动注入

**文件位置**：`hpd_agent/di/`

---

## 四、数据模型

### 4.1 Task（任务模型）

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | str | 任务唯一标识 |
| `name` | str | 任务名称 |
| `type` | TaskType | 任务类型 (SIMPLE/COMPLEX/EXPERT) |
| `difficulty` | float | 难度评分 [0,1] |
| `priority` | int | 优先级 [0,10] |
| `status` | TaskStatus | 状态 |
| `dependencies` | List[str] | 依赖任务ID列表 |
| `payload` | Dict | 任务负载数据 |
| `result` | Any | 执行结果 |
| `error` | str | 错误信息 |
| `start_time` | datetime | 开始时间 |
| `end_time` | datetime | 结束时间 |

### 4.2 RouteDecision（路由决策）

| 字段 | 类型 | 说明 |
|------|------|------|
| `node_id` | str | 目标节点ID |
| `node_name` | str | 目标节点名称 |
| `level` | RouteLevel | 路由层级 |
| `confidence` | float | 置信度 |

### 4.3 GenerationResult（生成结果）

| 字段 | 类型 | 说明 |
|------|------|------|
| `content` | str | 生成内容 |
| `score` | float | 评分 |
| `latency` | float | 延迟(秒) |
| `token_count` | int | Token数量 |

---

## 五、配置与部署

### 5.1 配置项说明

**配置文件**：`hpd_agent/config.py` / `.env`

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `llm_provider` | str | openai | LLM提供商 |
| `default_model` | str | gpt-4 | 默认模型 |
| `llm_temperature` | float | 0.7 | 温度参数 |
| `llm_max_tokens` | int | 4096 | 最大Token数 |
| `max_workers` | int | 10 | 并行工作线程数 |
| `timeout` | int | 300 | 超时时间(秒) |
| `max_retries` | int | 3 | 重试次数 |
| `cache_backend` | str | memory | 缓存后端 |
| `cache_ttl_seconds` | int | 3600 | 缓存TTL |

### 5.2 环境变量

| 变量名 | 说明 |
|--------|------|
| `OPENAI_API_KEY` | OpenAI API密钥 |
| `DASHSCOPE_API_KEY` | 阿里云DashScope密钥 |
| `ANTHROPIC_API_KEY` | Anthropic API密钥 |
| `GOOGLE_API_KEY` | Google API密钥 |
| `DEEPSEEK_API_KEY` | DeepSeek API密钥 |
| `REDIS_URL` | Redis连接地址 |

---

## 六、API接口

### 6.1 CLI接口

```bash
# 基本使用
python -m hpd_agent.cli --query "你的问题" --provider deepseek --model deepseek-chat

# 完整参数
python -m hpd_agent.cli --query "问题" \
  --provider openai \
  --model gpt-4 \
  --temperature 0.7 \
  --max-tokens 4096 \
  --max-workers 10 \
  --timeout 300 \
  --log-level INFO \
  --output-json
```

### 6.2 Python API

```python
from hpd_agent import HPDLLMAgent
from hpd_agent.config import AgentConfig

# 创建配置
config = AgentConfig(
    llm_provider="deepseek",
    default_model="deepseek-chat",
    max_workers=10,
    timeout=300
)

# 创建Agent
agent = HPDLLMAgent(config)

# 执行任务
result = await agent.run("分析气候变化对全球粮食安全的影响")

# 结果结构
{
    "success": True,
    "answer": "分析结果...",
    "score": 0.85,
    "route_level": "level_2",
    "node": "Complex Node",
    "task_count": 3,
    "success_count": 3
}
```

---

## 七、测试覆盖

### 7.1 测试文件结构

| 测试文件 | 覆盖模块 | 测试数量 |
|----------|----------|----------|
| `test_cache.py` | 缓存模块 | 4 |
| `test_dag_scheduler.py` | DAG调度器 | 3 |
| `test_di.py` | 依赖注入 | 4 |
| `test_expert_executor.py` | 专家执行器 | 4 |
| `test_llm_factory.py` | LLM工厂 | 3 |
| `test_multimodal.py` | 多模态处理 | 4 |
| `test_router.py` | 路由模块 | 4 |
| `test_rag.py` | RAG检索模块 | 7 |

### 7.2 测试运行

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定测试
python -m pytest tests/test_dag_scheduler.py -v
```

---

## 八、技术栈

### 8.1 核心依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| langchain | >=0.1.0 | LLM编排框架 |
| langgraph | >=0.0.60 | 工作流图 |
| openai | >=1.0.0 | OpenAI API |
| dashscope | >=1.14.0 | 阿里云LLM |
| pydantic | >=2.0.0 | 数据验证 |
| pydantic-settings | >=2.0.0 | 配置管理 |
| aiohttp | >=3.9.0 | 异步HTTP |
| pytest | >=7.0.0 | 测试框架 |

### 8.2 语言与环境

- **语言**: Python 3.11+
- **异步支持**: asyncio
- **类型提示**: PEP 484

---

## 九、扩展能力

### 9.1 插件化架构

系统采用插件化设计，支持：
- 新增LLM提供商插件
- 新增缓存后端插件
- 新增路由策略插件

### 9.2 监控与可观测性

预留扩展点：
- 任务执行指标收集
- 性能监控仪表盘
- 日志聚合与分析

---

## 十、安全与合规

### 10.1 安全特性

| 特性 | 实现 |
|------|------|
| API密钥管理 | 环境变量存储，不明文存储 |
| 输入验证 | Pydantic数据验证 |
| 异常处理 | 完整异常体系 |
| 日志脱敏 | 敏感信息脱敏处理 |

---

## 十一、典型使用场景

### 11.1 场景1：智能客服

```python
# 简单问题路由到Level 1
result = await agent.run("今天天气怎么样？")
# → 路由到 Simple Node (gpt-3.5-turbo)

# 复杂分析路由到Level 2
result = await agent.run("分析市场趋势并给出投资建议")
# → 路由到 Complex Node (gpt-4)

# 专业研究路由到Level 3
result = await agent.run("设计分布式系统架构")
# → 路由到 Expert Node (gpt-4-turbo)
```

### 11.2 场景2：多模态文档处理

```python
inputs = [
    MultimodalInput(type=InputType.TEXT, content="分析这份报告"),
    MultimodalInput(type=InputType.IMAGE, content=image_bytes),
    MultimodalInput(type=InputType.FILE, content=pdf_bytes)
]

result = await agent.run(inputs)
```

### 11.3 场景3：批量任务处理

```python
tasks = [
    {"name": "Task 1", "type": "simple", "dependencies": []},
    {"name": "Task 2", "type": "complex", "dependencies": ["t1"]},
    {"name": "Task 3", "type": "expert", "dependencies": ["t2"]}
]

dag = scheduler.build_dag_from_tasks(tasks)
results = await scheduler.schedule(dag, executor)
```

---

## 十二、版本历史

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| v0.1.0 | 2026-05 | 初始版本，核心功能实现 |
| v0.2.0 | 2026-05 | 性能优化升级：缓存、DI、多LLM支持 |

---

**文档版本**: v1.0  
**生成日期**: 2026-05-09  
**作者**: 灵智编排引擎团队

---

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/hpd-agent.git
cd hpd-agent

# 安装依赖
pip install -e .
```

### 配置

创建 `.env` 文件：

```env
OPENAI_API_KEY=your_api_key
DASHSCOPE_API_KEY=your_api_key
DEEPSEEK_API_KEY=your_api_key
MAX_WORKERS=10
TIMEOUT=300
LOG_LEVEL=INFO
```

### 运行

```bash
# CLI方式
python -m hpd_agent.cli --query "你好" --provider deepseek --model deepseek-chat

# Python API方式
python examples/basic_usage.py
```
