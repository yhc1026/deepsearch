# DeepSearch Agents

## 环境要求

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Docker & Docker Compose
- Node.js & [pnpm](https://pnpm.io/)
- 大模型 API Key（OpenAI 兼容接口）
- Tavily API Key
- RAGFlow 服务（可选，不使用知识库功能可跳过）

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 API Key 和服务地址：

```bash
# LLM
MODEL=qwen-max
API_KEY=你的API_KEY
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# Tavily 搜索
TAVILY_API_KEY=你的TAVILY_API_KEY

# RAGFlow（docker compose up -d 后可用）
RAGFLOW_API_URL=http://localhost:9380
RAGFLOW_API_KEY=ragflow-your-api-key

# MySQL（docker compose up -d 后可用）
MYSQL_USER=root
MYSQL_PASSWORD=root
MYSQL_DATABASE=deepsearch_db
MYSQL_HOST=localhost
MYSQL_PORT=3306
```

### 3. 启动基础设施

```bash
# MySQL（业务数据库）
cd docker && docker compose up -d

# RAGFlow 全家桶（知识库引擎 + Redis + MinIO + 向量库）
cd ragflow/docker && docker compose up -d
```

### 4. 启动后端服务

```bash
# 生产模式
uv run python start_services.py

# 开发模式（热重载）
uv run python start_services.py --reload
```

启动后各服务监听端口：

| 端口 | 服务 | 模块路径 |
|------|------|---------|
| 8000 | 主智能体（Orchestrator） | `agents.orchestrator.server:app` |
| 8001 | 网络搜索 Agent | `agents.network_search.server:app` |
| 8002 | 数据库查询 Agent | `agents.database_query.server:app` |
| 8003 | RAGFlow 知识库 Agent | `agents.ragflow_search.server:app` |
| 8100 | MySQL MCP Server | `agents.database_query.mysql_mcp_server:http_app` |

健康检查：`curl http://localhost:800{0,1,2,3}/health`

### 5. 启动前端

```bash
cd frontend
pnpm install
pnpm dev
```

## API 接口

主服务 (port 8000)：

| 接口 | 说明 |
|------|------|
| `POST /api/task` | 提交搜索任务 |
| `POST /api/task/{thread_id}/cancel` | 取消任务 |
| `POST /api/upload` | 上传附件 |
| `GET /api/files?path=...` | 列出输出文件 |
| `GET /api/download?path=...` | 下载文件 |
| `WebSocket /ws/{thread_id}` | 实时推送执行进度 |

## 项目结构

```
deepsearch/
├── agents/
│   ├── orchestrator/           # 主智能体（planner + executor + API server）
│   ├── network_search/         # 网络搜索 Agent（Tavily）
│   ├── database_query/         # 数据库查询 Agent（MySQL + MCP）
│   └── ragflow_search/         # RAGFlow 知识库 Agent
├── shared/                     # 公共基础库（llm / logger / monitor / A2A 基类）
├── docker/                     # Docker Compose 一键部署（MySQL + RAGFlow）
├── frontend/                   # React 前端
├── ragflow/                    # RAGFlow 服务端源码（仅参考，不参与构建）
├── start_services.py           # 一键启动所有 Agent 服务
├── pyproject.toml              # 项目依赖与配置
└── uv.lock                     # 依赖版本锁定
```

## 分布式部署

每个 Agent 只包含自己的依赖，可独立部署到不同机器：

```bash
# 只部署网络搜索 Agent
uv sync --group network-search
uv run uvicorn agents.network_search.server:app --host 0.0.0.0 --port 8001

# 只部署数据库查询 Agent
uv sync --group database-query
uv run uvicorn agents.database_query.server:app --host 0.0.0.0 --port 8002

# 只部署 RAGFlow 知识库 Agent
uv sync --group ragflow
uv run uvicorn agents.ragflow_search.server:app --host 0.0.0.0 --port 8003

# 只部署主智能体
uv sync --group orchestrator
uv run uvicorn agents.orchestrator.server:app --host 0.0.0.0 --port 8000
```

分布式部署时，修改 `agents/orchestrator/a2a_tools.py` 中的 `SUBAGENT_URLS` 指向各 Agent 的实际地址即可。
