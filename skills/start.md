# DeepSearch 项目启动指南

## 启动前检查

在开始之前，请确认以下三个前置条件已经满足：

1. Docker Desktop 处于运行状态——在终端执行 `docker ps` 能正常返回容器列表而不报错，即代表可用
2. 项目根目录下的 `.env` 文件存在，并且 `OPENAI_API_KEY`、`TAVILY_API_KEY`、`RAGFLOW_API_KEY` 都已填写正确的密钥
3. Windows 系统环境变量中不要存在同名的 `OPENAI_API_KEY`，否则会覆盖 `.env` 里的配置（之前的 401 认证错误就是这个原因）

## 第一步：激活虚拟环境

打开一个新的终端，切换到项目根目录并激活 conda 环境：

```bash
cd D:/code/programs/deepsearch
conda activate agent
```

后续所有后端相关的命令都必须在 `agent` 环境下执行。如果还没有创建这个环境，先用 `conda create -n agent python=3.12` 创建，再用 `uv sync` 安装所有依赖。

## 第二步：启动本地 MySQL

项目需要 MySQL 来存储业务数据（药品、库存、销售等）。通过 Docker Compose 一键启动：

```bash
docker compose -f docker/docker-compose.yaml up -d
```

执行完后需要等待 MySQL 初始化完毕。用下面这条命令监控状态，看到 `healthy` 再继续：

```bash
docker inspect deepsearch-mysql --format='{{.State.Health.Status}}'
```

## 第三步：启动 RAGFlow 知识库服务

RAGFlow 是知识库检索的后端，它依赖 Elasticsearch、MinIO、Redis 等多个容器，启动时间比较长。

首先进入 RAGFlow 的 Docker 目录并拉起所有服务：

```bash
cd ragflow/docker
docker compose up -d
```

然后**务必回到项目根目录**，否则后续步骤会在错误路径下执行：

```bash
cd D:/code/programs/deepsearch
```

RAGFlow 的核心是 Elasticsearch 容器，必须等它变为 healthy 才能正常提供服务。用下面命令持续检查，直到输出 `healthy`：

```bash
docker inspect docker-es01-1 --format='{{.State.Health.Status}}'
```

如果 ES 一直停留在 `health: starting` 甚至反复重启（退出码 137），说明内存不足被 Docker 杀掉了。解决方法：打开 Docker Desktop 设置 → Resources → 把内存调到 8GB 以上，或者编辑 `ragflow/docker/.env` 把 `MEM_LIMIT` 从 8GB 降到 2GB。

最后验证 RAGFlow API 是否正常响应：

```bash
curl -s -H "Authorization: Bearer ragflow-DWHqXEFvqbtm3rfQ7LODWZzswpRh25QX0IGFRyaKuLk" http://localhost:9381/api/v1/chats
```

看到返回 JSON 数据（即使 chats 为空）就说明 RAGFlow 就绪了。

## 第四步：启动后端 FastAPI 服务

在项目根目录下，用 uv 启动 Uvicorn：

```bash
uv run uvicorn app.api.server:app --host 0.0.0.0 --port 8000 --reload
```

启动成功后终端会显示 `Uvicorn running on http://0.0.0.0:8000`。

用 curl 验证后端是否正常响应：

```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/docs
```

返回 `200` 即表示后端启动成功。

常见问题：
- 如果报 GBK 编码错误，是因为 Windows 终端默认 GBK 无法处理 LLM 返回的 emoji，已在 `server.py` 中通过 `sys.stdout.reconfigure(encoding="utf-8")` 修复
- 如果报 401 认证错误，去 DeepSeek 平台检查 API Key 是否有效，并确认系统环境变量没有冲突

## 第五步：启动前端

另开一个新的终端窗口，进入前端目录并启动：

```bash
cd D:/code/programs/deepsearch/frontend
pnpm dev
```

启动后会显示本地访问地址，通常是 `http://localhost:5173`。

## 最终验证

所有服务都启动后，用以下命令逐一验证，每条都应返回 HTTP 200：

```bash
# 后端 API
curl -s -o /dev/null -w "后端: %{http_code}\n" http://localhost:8000/docs

# 前端界面
curl -s -o /dev/null -w "前端: %{http_code}\n" http://localhost:5173

# RAGFlow
curl -s -o /dev/null -w "RAGFlow: %{http_code}\n" http://localhost:9381

# MySQL（用 docker inspect 检查健康状态）
docker inspect deepsearch-mysql --format='MySQL: {{.State.Health.Status}}'
```

全部通过后，向后端发送一个测试任务验证端到端链路是否正常：

```bash
curl -s -X POST http://localhost:8000/api/task \
  -H "Content-Type: application/json" \
  -d '{"query": "用网络搜索查询今天的科技新闻，生成一份简短的 Markdown 报告", "thread_id": "smoke-test-001"}'
```

如果返回 200，且 WebSocket 连接正常推送进度、子智能体有调用记录、最终能生成报告，则启动完全成功。
