# MySQL Agent MCP 改造文档

## 1. 改造背景

DeepSearch 项目的数据库查询智能体（port 8002）原先通过 `mysql-connector-python` 直连 MySQL 执行查询。虽然功能完整，但存在以下问题：

- 工具与数据库之间是强耦合的本地函数调用，更换数据库或调整连接方式需要修改 Agent 代码
- 三个工具（`list_sql_tables`、`get_table_data`、`execute_sql_query`）无法被其他 Agent 或外部系统复用
- 不符合当前 AI Agent 工具标准化的趋势

因此引入 **MCP（Model Context Protocol）** 协议重构 MySQL 工具的调用链路，将工具逻辑与调用协议解耦。

---

## 2. 改造前架构

```
数据库查询智能体 (port 8002)
  │
  └─ LLM (DeepSeek / Qwen)
       │
       ├─ list_sql_tables()         ─── mysql.connector.connect() ─── MySQL
       ├─ get_table_data(table)     ─── mysql.connector.connect() ─── MySQL
       └─ execute_sql_query(sql)    ─── mysql.connector.connect() ─── MySQL

所有工具 = 本地 Python 函数
调用方式 = 进程内直接 import → 函数调用
```

**特点**：
- 三个 LangChain `@tool` 装饰的函数，进程内直接调用
- 工具与 `mysql.connector` 强耦合
- 数据库配置通过 `.env` + `load_dotenv()` 读取
- 蓝色日志打印在工具函数内部

---

## 3. 改造后架构

```
数据库查询智能体 (port 8002)
  │
  ├─ [优先] MCP 工具 (通过 stdio 子进程)
  │    │
  │    └─ MySQL MCP Server (uv run python -m app.mcp.mysql_mcp_server)
  │         │
  │         ├─ mysql_list_tables()   ─── mysql.connector ─── MySQL
  │         ├─ mysql_get_schema(t)   ─── mysql.connector ─── MySQL
  │         └─ mysql_execute(sql)    ─── mysql.connector ─── MySQL
  │
  │    协议: JSON-RPC over stdio
  │    工具加载: MCP Client → stdio_client() → ClientSession → list_tools()
  │    工具包装: 读取 inputSchema → 动态生成 Pydantic Model → StructuredTool
  │
  └─ [兜底] 直连工具 (MCP 不可用时自动回退)
       │
       ├─ list_sql_tables()          ─── mysql.connector ─── MySQL
       ├─ get_table_data(table)      ─── mysql.connector ─── MySQL
       └─ execute_sql_query(sql)     ─── mysql.connector ─── MySQL
```

**核心变化**：

| 维度 | 改造前 | 改造后 |
|------|--------|--------|
| 调用协议 | Python 函数调用 | MCP JSON-RPC over stdio |
| 工具注册 | `@tool` 装饰器 | `server.add_request_handler("tools/list", ...)` |
| 进程模型 | 单进程 | Agent 进程 + MCP Server 子进程 |
| 工具来源 | 静态 import | 动态从 MCP Server 发现 + 兜底代码 |
| 可复用性 | 仅限本进程 | 任何 MCP Client 均可连接 |
| 容错性 | 无降级 | MCP 不可用时自动回退直连工具 |

---

## 4. 新增/修改文件一览

### 新建文件

#### `app/mcp/mysql_mcp_server.py` — MCP Server

```python
# MCP 2.0 标准 Server，stdio transport
server = Server("MySQL MCP Server")

# 注册 tools/list 处理器：声明 3 个工具
server.add_request_handler("tools/list", ListToolsRequest, handle_list_tools)

# 注册 tools/call 处理器：按工具名分发执行
server.add_request_handler("tools/call", CallToolRequest, handle_call_tool)

# 工具逻辑与 db_tools.py 完全一致
# 蓝色日志打印保留：
#   [MySQL MCP] 查询数据库表名: SHOW TABLES        # 蓝色
#   [MySQL MCP] 预览表数据: SELECT * FROM xxx ...  # 蓝色
#   [MySQL MCP] 执行 SQL 查询: <sql>               # 蓝色
```

#### `app/mcp/client.py` — MCP Client 加载器

```python
# 通过 stdio 连接 MCP Server 子进程
# 自动读取工具定义的 inputSchema，生成 Pydantic 参数模型
# 将 MCP 工具包装为 LangChain StructuredTool

async def _load_mcp_tools_async():
    params = StdioServerParameters(
        command="uv",
        args=["run", "python", "-m", "app.mcp.mysql_mcp_server"],
    )
    # ... 连接、初始化、list_tools、包装为 StructuredTool
    return tools, exit_stack

# 模块导入时自动加载
MYSQL_MCP_TOOLS = asyncio.run(_load_mcp_tools_async())
```

#### `app/mcp/__init__.py` — 包初始化

### 修改文件

#### `app/services/database_query_service.py`

```python
# 改造前
from app.tools.db_tools import list_sql_tables, get_table_data, execute_sql_query
tools = [list_sql_tables, get_table_data, execute_sql_query]

# 改造后
from app.mcp.client import MYSQL_MCP_TOOLS       # MCP 工具（优先）
from app.tools.db_tools import (                  # 直连工具（兜底）
    list_sql_tables, get_table_data, execute_sql_query
)

if MCP 加载成功:
    tools = MCP工具列表 + 兜底工具列表
    system_prompt += MCP工具说明 + "优先用MCP，不够再兜底"
else:
    tools = 兜底工具列表  # 服务不受影响
    print("MCP 加载失败，使用直连工具")
```

#### `pyproject.toml`

```diff
   "deepagents==0.5.7",
+  "langchain-mcp-adapters>=0.1.0",
+  "mcp>=1.0.0",
```

---

## 5. 改进点与成果

### 5.1 协议标准化

工具从进程内函数调用升级为 MCP JSON-RPC 标准协议。MySQL MCP Server 可以被任何实现了 MCP Client 的系统连接，不再局限于本项目。

### 5.2 热插拔与容错

- MCP Server 正常时：优先使用 MCP 工具（标准化路径）
- MCP Server 异常时：自动回退到直连工具（`list_sql_tables` / `get_table_data` / `execute_sql_query`）
- Agent LLM 通过动态注入的提示词自行判断何时使用 MCP 工具、何时回退

### 5.3 工具发现自动化

MCP Client 通过 `tools/list` 协议动态发现工具定义（名称、描述、参数 schema），不需要在 Agent 端硬编码工具签名。添加新工具只需在 MCP Server 端注册，Agent 端无需修改。

### 5.4 进程隔离

MCP Server 作为独立子进程运行，崩溃不影响 Agent 主进程。子进程由 `AsyncExitStack` 管理生命周期，Agent 进程退出时自动清理。

### 5.5 为后续扩展打下基础

本改造验证了 MCP 协议在 DeepSearch 项目中的可行性。后续可以：

- 将 Tavily 搜索工具改造为 MCP Server
- 将 RAGFlow 知识库工具改造为 MCP Server
- 接入社区已有的 GitHub / Slack / Filesystem 等 MCP Server
- 最终将所有子智能体收敛到统一的 MCP 管家架构

### 5.6 日志可观测性不变

蓝色 SQL/查询日志从工具函数内部同步迁移到 MCP Server 端，效果完全一致：

```
============================================================
[MySQL MCP] 执行 SQL 查询:
  SELECT drug_name, SUM(quantity) FROM inventory GROUP BY drug_name ORDER BY SUM(quantity) DESC
============================================================
```

---

## 6. 技术要点

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| MCP SDK | `mcp` 2.0.0 | 官方 Python SDK，支持 stdio/SSE transport |
| Server API | `mcp.server.Server` + `add_request_handler` | MCP 2.0 标准模式，手动注册 tools/list 和 tools/call |
| Client 连接 | `mcp.StdioServerParameters` + `stdio_client` | 子进程 stdio transport |
| LangChain 集成 | `langchain-mcp-adapters` + 自研 `StructuredTool` 包装 | 动态解析 inputSchema，生成 Pydantic 参数模型 |
| 协议序列化 | JSON-RPC 2.0 | MCP 底层协议 |

---

## 7. 自定义 Client 与 langchain-mcp-adapters 的权衡

### 7.1 两种方案对比

当前 `app/mcp/client.py` 使用 MCP SDK 原生 API 自行实现客户端加载逻辑。项目同时安装了 `langchain-mcp-adapters` 0.3.1，它提供了开箱即用的封装。两者的取舍如下：

| 维度 | 自定义 Client（当前方案） | `langchain-mcp-adapters` |
|------|--------------------------|---------------------------|
| 代码量 | ~120 行 | ~10 行 |
| 依赖 | 仅 `mcp` SDK | `mcp` + `langchain-mcp-adapters` |
| MCP 2.0 兼容 | 已验证，直接适配 | 声称支持，未实测验证 |
| 参数模型 | 从 `inputSchema` 动态生成 Pydantic Model，LLM 看到结构化参数 | 不保证生成 `args_schema`，可能退化为 `**kwargs` |
| 调试可控性 | 每行代码自己掌握，出错直接定位 | 出错需排查适配层逻辑 |
| 子进程管理 | `AsyncExitStack` 显式控制生命周期 | 适配器内部封装，行为不透明 |
| 扩展定制 | 完全自由 | 受适配器 API 约束 |

### 7.2 选择自定义方案的理由

**MCP 2.0 Breaking Changes**

MCP 2.0 相比 1.x 删除了 `FastMCP` 装饰器 API，`add_request_handler` 签名从 `(method, handler)` 变为 `(method, params_type, handler)`，handler 被调用时传入 `(ctx, params)` 两个参数而不是一个。在 Server 端已验证了这些变化，但 `langchain-mcp-adapters` 0.3.1 是否完全适配了这些变更未经实测。与其花时间排查第三方适配层的问题，不如用原生 API 一次写对。

**Pydantic 参数模型**

这是决定性的差异。MCP Server 端定义了每个工具的 `inputSchema`（JSON Schema 格式）：

```json
{
  "name": "mysql_get_schema",
  "inputSchema": {
    "type": "object",
    "properties": {
      "table_name": {
        "type": "string",
        "description": "要预览的表名"
      }
    },
    "required": ["table_name"]
  }
}
```

`client.py` 在加载工具时读取 `inputSchema`，动态生成 Pydantic 模型：

```python
args_model = create_model("mysql_get_schema_args",
    table_name=(str, Field(description="要预览的表名")),
)
tool = StructuredTool.from_function(..., args_schema=args_model)
```

这保证了 LLM 看到的工具签名包含**类型信息**和**字段描述**，参数填写准确率显著高于没有 schema 的情况。`langchain-mcp-adapters` 不保证做这一步——如果它跳过 `args_schema`，LLM 只能用 `**kwargs` 自由发挥，容易出现参数名拼错、类型不匹配等问题。

### 7.3 如果改用适配器的代码

```python
# client.py 可从 ~120 行精简到 ~20 行
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

async def _load():
    client = MultiServerMCPClient({
        "mysql": {
            "transport": "stdio",
            "command": "uv",
            "args": ["run", "python", "-m", "app.mcp.mysql_mcp_server"],
        }
    })
    return await client.get_tools()

MYSQL_MCP_TOOLS = asyncio.run(_load())
```

**迁移前提**：需验证 `get_tools()` 返回的 LangChain 工具是否携带正确的 `args_schema`。如果带，切换成本为零；如果不带，当前自定义方案更优。

### 7.4 结论

- **当前阶段**（MCP 2.0 刚发布、适配器生态未成熟）：自定义 Client 更可控，Pydantic 参数模型保证 LLM 调用质量
- **后续阶段**（适配器稳定、社区验证充分）：可以切到 `MultiServerMCPClient`，减少维护负担
- **不影响 Server 端**：无论 Client 用哪种方案，`mysql_mcp_server.py` 保持不变，因为它是标准 MCP Server，任何 MCP Client 都能连接

---

## 8. 启动方式

无需改变启动命令。`start_services.py` 照常启动所有服务，MySQL MCP Server 由 `database_query_service.py` 在导入 `app.mcp.client` 时自动作为子进程拉起。

```bash
uv run python start_services.py
# 或
uv run python start_services.py --reload
```

如需独立测试 MCP Server：

```bash
uv run python -m app.mcp.mysql_mcp_server
```
