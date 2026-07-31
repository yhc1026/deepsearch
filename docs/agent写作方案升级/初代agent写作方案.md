# DeepSearch Agents 协作详细清单

> 备份日期：2026-07-31
> 用途：拆分前的架构快照，记录当前 1 主 + 3 从智能体的完整协作关系
> 目标：将 4 个 Agent 解耦为独立进程 → 分布式部署

---

## 一、架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│  port 8000: FastAPI (app.api.server:app)                        │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  main_agent (DeepAgent)                                   │  │
│  │  model: qwen-max (via OpenAI compatible API)              │  │
│  │  checkpointer: InMemorySaver()                            │  │
│  │  system_prompt: prompts.yml → main_agent.system_prompt    │  │
│  │                                                           │  │
│  │  tools (主智能体直接持有，共 3 个):                        │  │
│  │    ├── generate_markdown     (app/tools/markdown_tools.py) │  │
│  │    ├── convert_md_to_pdf     (app/tools/pdf_tools.py)     │  │
│  │    └── read_file_content     (app/tools/upload_file_read_tool.py)│
│  │                                                           │  │
│  │  subagents (字典式子智能体，同进程内执行，共 3 个):         │  │
│  │    ├── 网络搜索助手 (network_search_agent)                 │  │
│  │    ├── 数据库查询助手 (database_query_agent)               │  │
│  │    └── RAGFlow助手 (knowledge_base_agent)                 │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  HTTP Endpoints:                                                │
│    POST /api/task                   启动 Agent 任务              │
│    POST /api/task/{thread_id}/cancel  取消任务                   │
│    POST /api/upload                  上传文件                    │
│    GET  /api/files                   列出生成文件                │
│    GET  /api/download                下载文件                    │
│    WS   /ws/{thread_id}             实时推送执行事件             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、Agent 详细信息

### 2.1 主智能体 (Main Agent)

| 项目 | 详情 |
|------|------|
| **角色** | 金融与电商研究团队负责人 |
| **模型** | `qwen-max`，通过 OpenAI 兼容 API 接入 |
| **checkpointer** | `InMemorySaver`，按 `thread_id` 隔离会话记忆 |
| **创建方式** | `deepagents.create_deep_agent()` |
| **文件位置** | `app/agent/main_agent.py` |

**职责：**
1. 理解用户任务，规划执行步骤
2. 按需分派子智能体：网络搜索、数据库查询、RAGFlow 知识库
3. 汇总子智能体返回的信息
4. 调用文件工具生成 Markdown / PDF 交付物
5. 通过 monitor + WebSocket 向同一 thread_id 的前端推送进度

**持有工具 (3 个)：**

| 工具名 | 文件 | 功能 | 依赖 |
|--------|------|------|------|
| `generate_markdown` | `app/tools/markdown_tools.py` | 将文本写入 .md 文件到当前会话目录 | `resolve_path`、`ContextVar(session_dir)` |
| `convert_md_to_pdf` | `app/tools/pdf_tools.py` | 将 .md 文件转换为 PDF | `resolve_path`、`word_converter.convert_md_to_pdf_via_word` |
| `read_file_content` | `app/tools/upload_file_read_tool.py` | 读取上传附件（.md/.txt/.docx/.pdf/.xlsx） | `pypdf`、`python-docx`、`pandas`、`ContextVar(session_dir)` |

**持有的子智能体 (3 个字典式定义)：**
- `app/agent/subagents/network_search_agent.py` → `network_search_agent`
- `app/agent/subagents/database_query_agent.py` → `database_query_agent`
- `app/agent/subagents/knowledge_base_agent.py` → `knowledge_base_agent`

---

### 2.2 网络搜索助手 (Network Search Agent)

| 项目 | 详情 |
|------|------|
| **角色** | 互联网公开信息检索 |
| **文件位置** | `app/agent/subagents/network_search_agent.py` |
| **工具** | `internet_search` (app/tools/tavily_tool.py) |
| **描述** | "负责进行网络知识搜索的智能体助手，当需要从网络中查询数据的时候，可以执行数据检索" |

**调用链路：**
```
主智能体 task(description="搜索某问题")
  → DeepAgents 调用 network_search_agent
  → internet_search(query, topic, max_results, include_raw_content)
  → Tavily API 搜索
  → 返回结构化搜索结果
```

**system_prompt 约束：**
- 至少检索 3 个角度的问题
- 最多 5 次检索
- 超过 5 次不允许继续检索

**internet_search 工具参数：**
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | str | 必填 | 搜索关键词或问题 |
| `topic` | Literal["news","finance","general"] | "general" | 搜索主题 |
| `max_results` | int | 5 | 最大返回结果数 |
| `include_raw_content` | bool | False | 是否返回网页原文 |

**外部依赖：**
- Tavily API (`TAVILY_API_KEY` in .env)
- 网络请求到 `https://api.tavily.com`

---

### 2.3 数据库查询助手 (Database Query Agent)

| 项目 | 详情 |
|------|------|
| **角色** | MySQL 结构化数据查询 |
| **文件位置** | `app/agent/subagents/database_query_agent.py` |
| **工具** | `list_sql_tables`, `get_table_data`, `execute_sql_query` (app/tools/db_tools.py) |
| **描述** | "负责进行数据库查询的智能体助手。它可以查看数据库中有哪些表，读取表数据和查看表结构，并执行自定义SQL查询" |

**调用链路：**
```
主智能体 task(description="查询某药品库存")
  → DeepAgents 调用 database_query_agent
  → 三步工作流:
     1. list_sql_tables()           → 返回表名列表
     2. get_table_data(table_name)  → 返回表结构 + 前100行数据
     3. execute_sql_query(query)    → 执行自定义 SQL
  → 返回 CSV 格式查询结果
```

**三个工具详情：**

| 工具 | SQL 操作 | 返回格式 | 用途 |
|------|----------|----------|------|
| `list_sql_tables` | `SHOW TABLES` | "可用的表有：t1, t2, t3" | 发现表名 |
| `get_table_data` | `SELECT * FROM {table} LIMIT 100` | CSV (列名行 + 数据行) | 预览结构 |
| `execute_sql_query` | 任意 SELECT/SHOW | CSV (列名行 + 数据行) | 自定义查询 |

**DB 连接参数（从 .env 读取）：**
- `MYSQL_HOST` (default: localhost)
- `MYSQL_PORT` (default: 3306, docker 建议 3307)
- `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`
- `MYSQL_CHARSET=utf8mb4`, `MYSQL_COLLATION=utf8mb4_unicode_ci`

**数据库（deepsearch_db）：**
- `drugs` — 50 种药品主数据（名称、规格、厂家、治疗领域）
- `inventory` — 150 条库存记录（批号、数量、仓库、效期），FK→drugs
- `sales_records` — 100 条销售记录（日期、数量、金额、客户、区域），FK→drugs

**外部依赖：**
- `mysql-connector-python`
- Docker MySQL 8.4 容器（docker/docker-compose.yaml）
- `docker/mysql/mysql.sql` 初始化数据

---

### 2.4 RAGFlow 助手 (Knowledge Base Agent)

| 项目 | 详情 |
|------|------|
| **角色** | 企业内部非结构化文档查询 |
| **文件位置** | `app/agent/subagents/knowledge_base_agent.py` |
| **工具** | `get_assistant_list`, `create_ask_delete` (app/tools/ragflow_tools.py) |
| **描述** | "负责查询 RAGFlow 内部知识库中的非结构化文档信息，例如 PDF、白皮书、研报、制度文件、产品资料等" |

**调用链路：**
```
主智能体 task(description="查询知识库中的某信息")
  → DeepAgents 调用 knowledge_base_agent
  → 两步工作流:
     1. get_assistant_list()                        → 返回可用助手及其知识库
     2. create_ask_delete(chat_name, question)       → 创建临时会话 → 提问 → 返回回答 → 删除会话
  → 返回 RAGFlow 回答文本
```

**两个工具详情：**

| 工具 | RAGFlow API 操作 | 说明 |
|------|-----------------|------|
| `get_assistant_list` | `ragflow_client.list_chats()` | 返回每个助手的名称、描述、关联知识库 |
| `create_ask_delete` | create session → completions (SSE流式) → delete session | 临时会话问答，用完即删 |

**system_prompt 约束：**
- 必须先调用 `get_assistant_list` 再提问
- 复杂问题至少 3 个角度提问；简单问题可只提 1 次
- 不合适的助手不强用，直接说"没有可用知识库"
- 返回保留原始回答、来源，不做最终结论（交给主智能体汇总）

**外部依赖：**
- RAGFlow SDK (`ragflow-sdk`)
- `RAGFLOW_API_URL`, `RAGFLOW_API_KEY` in .env
- RAGFlow 服务（独立部署，不在本项目 docker-compose 中）

---

## 三、跨智能体协作流程

### 3.1 典型任务执行流程

```
用户请求（HTTP POST /api/task）
  │
  ▼
run_deep_agent(task_query, session_id)
  │
  ├─ 创建会话目录: output/session_{session_id}/
  ├─ 复制上传文件: updated/{session_id}/* → output/{session_id}/
  ├─ 设置 ContextVar: session_dir, thread_id
  ├─ 注入工作环境指令 path_instruction
  │
  ▼
main_agent.astream({"messages": [user_msg + path_instruction]}, config)
  │
  ├─ [模型输出 tool_calls]
  │   │
  │   ├─ name="task", subagent_type="网络搜索助手"
  │   │   → DeepAgents 内部调 network_search_agent
  │   │   → internet_search → Tavily API
  │   │   → 返回搜索结果给主智能体
  │   │
  │   ├─ name="task", subagent_type="数据库查询助手"
  │   │   → DeepAgents 内部调 database_query_agent
  │   │   → list_sql_tables → get_table_data → execute_sql_query
  │   │   → 返回 CSV 数据给主智能体
  │   │
  │   ├─ name="task", subagent_type="RAGFlow助手"
  │   │   → DeepAgents 内部调 knowledge_base_agent
  │   │   → get_assistant_list → create_ask_delete
  │   │   → 返回知识库回答给主智能体
  │   │
  │   ├─ name="generate_markdown"   → 主智能体直接调用
  │   ├─ name="convert_md_to_pdf"   → 主智能体直接调用
  │   └─ name="read_file_content"   → 主智能体直接调用
  │
  ▼
主智能体汇总 → monitor.report_task_result(final_answer)
  │
  ▼
WebSocket 推送 → 前端展示结果
```

### 3.2 主智能体→子智能体的消息格式

当前通过 DeepAgents 内部 `task` tool call，主智能体调用子智能体时传递：
```json
{
  "name": "task",
  "args": {
    "subagent_type": "网络搜索助手",  // 子智能体 name
    "description": "搜索 2026 AI 趋势"  // 任务描述
  }
}
```

子智能体返回的是字符串文本，拼接到主智能体的消息历史中。

### 3.3 上下文传递机制

| 机制 | 使用者 | 传递内容 |
|------|--------|----------|
| `ContextVar: session_dir` | 主智能体的文件工具 | 当前会话工作目录绝对路径 |
| `ContextVar: thread_id` | monitor | WebSocket 路由 key |
| `config.thread_id` | DeepAgents checkpointer | 同一会话的 LLM 对话历史 |
| `path_instruction` (注入到用户消息) | 所有 Agent | 工作目录路径 + 上传文件列表 |

---

## 四、事件监控与前端通信

### 4.1 Monitor 事件类型

| 事件 | 触发位置 | 前端展示 |
|------|----------|----------|
| `session_created` | `run_deep_agent()` | 工作目录路径 |
| `tool_start` | 各工具内部 `monitor.report_tool()` | 工具名 + 参数 |
| `assistant_call` | `main_agent.astream` 循环中检测 `task` tool call | 子智能体名称 + description |
| `task_result` | `main_agent.astream` 中模型最终消息 | 最终回答文本 |
| `task_cancelled` | `run_deep_agent` CancelledError 分支 | 取消提示 |
| `error` | `run_deep_agent` Exception 分支 | 错误信息 |

### 4.2 WebSocket 连接管理

- `ConnectionManager` 维护 `thread_id → WebSocket` 映射
- FastAPI lifespan 启动时绑定事件循环
- monitor 通过 `ContextVar` 获取 thread_id，定向推送

---

## 五、依赖关系矩阵

```
                    ┌─────────────┐
                    │  .env 配置   │
                    └──────┬──────┘
           ┌───────────────┼───────────────┬───────────────┐
           ▼               ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
    │OpenAI API│   │Tavily API│   │  MySQL   │   │ RAGFlow  │
    │(qwen-max)│   │          │   │(Docker)  │   │ Server   │
    └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
    ┌────────────────────────────────────────────────────────┐
    │                    main_agent                          │
    │  + generate_markdown / convert_md_to_pdf (本地文件IO)  │
    │  + read_file_content (本地文件IO)                      │
    └────────────────────────────────────────────────────────┘
```

| 组件 | 外部依赖 | 网络访问 | 本地文件IO |
|------|----------|----------|------------|
| main_agent | OpenAI API | 是 | 是 (output/, updated/) |
| network_search_agent | Tavily API | 是 | 否 |
| database_query_agent | MySQL:3306 | 是 (localhost) | 否 |
| knowledge_base_agent | RAGFlow HTTP | 是 | 否 |

---

## 六、关键文件索引

| 文件 | 角色 |
|------|------|
| `app/api/server.py` | FastAPI 入口，HTTP/WS 接口 |
| `app/agent/main_agent.py` | 主智能体组装 + `run_deep_agent` 执行入口 |
| `app/agent/llm.py` | OpenAI 兼容模型初始化 |
| `app/agent/prompts.py` | YAML 提示词加载 |
| `app/prompt/prompts.yml` | 主/子智能体提示词配置 |
| `app/agent/subagents/network_search_agent.py` | 网络搜索助手定义 |
| `app/agent/subagents/database_query_agent.py` | 数据库查询助手定义 |
| `app/agent/subagents/knowledge_base_agent.py` | RAGFlow 助手定义 |
| `app/tools/tavily_tool.py` | Tavily 搜索工具 |
| `app/tools/db_tools.py` | MySQL 查询工具 (3个) |
| `app/tools/ragflow_tools.py` | RAGFlow 查询工具 (2个) |
| `app/tools/markdown_tools.py` | Markdown 生成工具 |
| `app/tools/pdf_tools.py` | PDF 转换工具 |
| `app/tools/upload_file_read_tool.py` | 上传文件读取工具 |
| `app/api/monitor.py` | ToolMonitor + ConnectionManager |
| `app/api/context.py` | ContextVar (session_dir, thread_id) |
| `app/utils/path_utils.py` | 路径解析/安全约束 |
| `app/utils/word_converter.py` | Markdown→PDF 底层转换 |
| `app/ragflow/rag_config.py` | RAGFlow 连接配置 |
| `docker/docker-compose.yaml` | MySQL 8.4 容器 |
| `docker/mysql/mysql.sql` | 数据库初始化 SQL |
| `.env.example` | 环境变量示例 |

---

## 七、拆分解耦要点 (预备)

以下是将要进行的架构变更：

1. **通信协议**: A2A (Agent-to-Agent)，HTTP REST + JSON
2. **子智能体独立化**: 每个子智能体变成独立 FastAPI 进程，暴露 `/invoke` 端点
3. **主智能体工具化**: 3 个子智能体变成 3 个 HTTP 包装工具
4. **checkpointer 独立**: 每个服务各自管理 `InMemorySaver`
5. **共享代码提取**: LLM、monitor(简化)、prompts → `app/shared/`
6. **统一启动脚本**: 一个脚本拉取 4 个服务
7. **端口分配**: 8000(主), 8001(网络搜索), 8002(数据库), 8003(RAGFlow)
