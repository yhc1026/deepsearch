# DeepSearch Agent 解耦实施计划

> 创建日期：2026-07-31
> 目标：将 1 主 + 3 从 Agent 从单进程 DeepAgents 子智能体模式，拆分为 4 个独立进程
> 第一步：单机拆分，部署在不同 localhost 端口
> 第二步（后续）：分布式部署到不同服务器

---

## 一、架构变更总览

### 改前（单进程）

```
port 8000: FastAPI
└── main_agent (DeepAgent)
    ├── tools: generate_markdown, convert_md_to_pdf, read_file_content
    └── subagents (DeepAgents 字典式，同进程):
        ├── network_search_agent  → tavily_tool
        ├── database_query_agent  → db_tools × 3
        └── knowledge_base_agent  → ragflow_tools × 2
```

### 改后（4 进程）

```
用户输入: "昨天天气如何？"
         │
         ▼
port 8000: 主智能体 (Orchestrator + Query Rewriter)
│
│  [Query 重写层]  ← 核心新增能力
│  "昨天" → "2026年7月30日"
│  "天气" → "气温 降水 湿度 风速 天气状况"
│  重写后: "2026年7月30日 北京 气温 降水 湿度 风速"
│
├── tools: generate_markdown, convert_md_to_pdf, read_file_content
└── tools: call_network_search(query="2026年7月30日 气温 降水...")
                   │
                   │  HTTP A2A POST /tasks
                   ▼
         port 8001: 网络搜索服务
         (DeepAgent + 自有 checkpointer)
         ├── internet_search
         └── 返回: "7月30日北京晴，气温32°C..."

其他子服务同理:
  port 8002: 数据库查询服务 (DeepAgent + MySQL tools)
  port 8003: RAGFlow 服务 (DeepAgent + RAGFlow tools)
```

---

## 二、A2A 协议设计

采用 Google A2A (Agent-to-Agent) 协议的简化版，基于 HTTP REST + JSON。

### 2.1 Agent Card（代理名片）

**端点**: `GET /`

每个 Agent 服务在根路径返回自身能力描述，供主智能体或其他服务发现。

```json
{
  "name": "网络搜索助手",
  "description": "负责进行网络知识搜索的智能体助手...",
  "version": "1.0.0",
  "capabilities": [
    {
      "name": "internet_search",
      "description": "根据关键词搜索互联网公开信息"
    }
  ],
  "endpoints": {
    "tasks": "/tasks",
    "health": "/health"
  }
}
```

### 2.2 Task（任务执行）

**端点**: `POST /tasks`

主智能体向子智能体服务发送任务，同步等待结果返回。

**Request:**
```json
{
  "task_id": "uuid-optional",
  "query": "搜索2026年AI在电商行业的应用趋势"
}
```

**Response (成功):**
```json
{
  "task_id": "uuid",
  "status": "completed",
  "result": "根据搜索结果，2026年AI在电商行业的应用趋势包括...",
  "agent_name": "网络搜索助手"
}
```

**Response (失败):**
```json
{
  "task_id": "uuid",
  "status": "failed",
  "error": "Tavily API 请求超时",
  "agent_name": "网络搜索助手"
}
```

### 2.3 Health Check（健康检查）

**端点**: `GET /health`

```json
{
  "status": "ok",
  "agent": "网络搜索助手"
}
```

### 2.4 主智能体侧的 A2A 工具

每个子智能体对应主智能体的一个 LangChain tool，工具内部发 HTTP POST 到对应服务的 `/tasks`：

```python
@tool
def call_network_search(query: str) -> str:
    """调用网络搜索助手，从互联网检索公开信息。
    适用场景：需要查询新闻、政策、行业趋势、网页资料等公开信息时使用。
    """
    response = requests.post(
        "http://localhost:8001/tasks",
        json={"query": query},
        timeout=120,
    )
    data = response.json()
    if data["status"] == "completed":
        return data["result"]
    return f"网络搜索助手返回错误: {data.get('error', '未知错误')}"
```

三个工具同样模式，仅 URL 和 docstring 不同：
- `call_network_search` → `http://localhost:8001/tasks`
- `call_database_query` → `http://localhost:8002/tasks`
- `call_ragflow_query` → `http://localhost:8003/tasks`

---

## 三、文件变更清单

### 3.1 新建文件

| # | 文件路径 | 用途 |
|---|----------|------|
| 1 | `app/shared/__init__.py` | 共享模块初始化 |
| 2 | `app/shared/llm.py` | 从 `app/agent/llm.py` 提取，供所有服务复用 |
| 3 | `app/shared/prompts.py` | 从 `app/agent/prompts.py` 提取，供所有服务复用 |
| 4 | `app/services/__init__.py` | 服务模块初始化 |
| 5 | `app/services/base.py` | A2A 服务基类：封装 FastAPI 创建、Agent Card、/tasks、/health |
| 6 | `app/services/network_search_service.py` | 网络搜索服务 (port 8001) |
| 7 | `app/services/database_query_service.py` | 数据库查询服务 (port 8002) |
| 8 | `app/services/ragflow_service.py` | RAGFlow 知识库服务 (port 8003) |
| 9 | `app/tools/a2a_agent_tools.py` | 主智能体用的 3 个 HTTP 包装工具 |
| 10 | `start_services.py` | 一键启动 4 个服务的 Python 脚本 |

### 3.2 修改文件

| # | 文件路径 | 变更内容 |
|---|----------|----------|
| 1 | `app/agent/main_agent.py` | 替换 subagents 为 A2A 工具；更新 import 路径 |
| 2 | `app/prompt/prompts.yml` | 更新主智能体 system_prompt，反映 tool 调用模式 |
| 3 | `app/agent/llm.py` | 改为 re-export from `app.shared.llm`（保持向后兼容） |
| 4 | `app/agent/prompts.py` | 改为 re-export from `app.shared.prompts`（保持向后兼容） |

### 3.3 不变文件

| 文件 | 说明 |
|------|------|
| `app/tools/tavily_tool.py` | 网络搜索服务直接引用 |
| `app/tools/db_tools.py` | 数据库查询服务直接引用 |
| `app/tools/ragflow_tools.py` | RAGFlow 服务直接引用 |
| `app/tools/markdown_tools.py` | 主智能体保留 |
| `app/tools/pdf_tools.py` | 主智能体保留 |
| `app/tools/upload_file_read_tool.py` | 主智能体保留 |
| `app/api/server.py` | 不变，仍调用 `run_deep_agent()` |
| `app/api/monitor.py` | 子服务不再需要 WebSocket 推送，仅主服务使用 |
| `app/api/context.py` | 不变 |
| `app/utils/*` | 不变 |
| `app/ragflow/*` | 不变 |
| `docker/*` | 不变 |

---

## 四、详细实现设计

### 4.1 A2A 服务基类 (`app/services/base.py`)

```python
class A2AAgentService:
    """A2A Agent 服务基类

    封装：
    1. FastAPI 应用创建
    2. Agent Card 端点 (GET /)
    3. 任务执行端点 (POST /tasks)
    4. 健康检查端点 (GET /health)
    5. DeepAgent 创建与执行

    子类只需提供 name, description, tools, system_prompt 即可。
    """

    def __init__(self, name, description, tools, system_prompt, port):
        self.name = name
        self.description = description
        self.tools = tools
        self.system_prompt = system_prompt
        self.port = port
        self.agent = None

    def create_agent(self):
        """创建本服务的 DeepAgent 实例（独立的 checkpointer）"""
        self.agent = create_deep_agent(
            model=model,
            system_prompt=self.system_prompt,
            tools=self.tools,
            checkpointer=InMemorySaver(),
        )

    def build_app(self) -> FastAPI:
        """构建 FastAPI 应用，注册 A2A 端点"""
        app = FastAPI(title=self.name)

        @app.get("/")
        async def agent_card():
            return {
                "name": self.name,
                "description": self.description,
                "version": "1.0.0",
                "capabilities": [...],
                "endpoints": {"tasks": "/tasks", "health": "/health"},
            }

        @app.post("/tasks")
        async def execute_task(request: TaskRequest):
            task_id = request.task_id or str(uuid.uuid4())
            try:
                result = await self.agent.ainvoke({
                    "messages": [{"role": "user", "content": request.query}]
                })
                # 提取最终消息
                final_msg = result["messages"][-1].content
                return {
                    "task_id": task_id,
                    "status": "completed",
                    "result": final_msg,
                    "agent_name": self.name,
                }
            except Exception as e:
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": str(e),
                    "agent_name": self.name,
                }

        @app.get("/health")
        async def health():
            return {"status": "ok", "agent": self.name}

        return app
```

### 4.2 网络搜索服务 (`app/services/network_search_service.py`)

```python
from app.shared.llm import model
from app.shared.prompts import sub_agents_content
from app.tools.tavily_tool import internet_search
from app.services.base import A2AAgentService

tavily_config = sub_agents_content["tavily"]

service = A2AAgentService(
    name=tavily_config["name"],
    description=tavily_config["description"],
    tools=[internet_search],
    system_prompt=tavily_config["system_prompt"],
    port=8001,
)

service.create_agent()
app = service.build_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

### 4.3 数据库查询服务 (`app/services/database_query_service.py`)

```python
from app.shared.llm import model
from app.shared.prompts import sub_agents_content
from app.tools.db_tools import list_sql_tables, get_table_data, execute_sql_query
from app.services.base import A2AAgentService

db_config = sub_agents_content["db"]

service = A2AAgentService(
    name=db_config["name"],
    description=db_config["description"],
    tools=[list_sql_tables, get_table_data, execute_sql_query],
    system_prompt=db_config["system_prompt"],
    port=8002,
)

service.create_agent()
app = service.build_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
```

### 4.4 RAGFlow 服务 (`app/services/ragflow_service.py`)

```python
from app.shared.llm import model
from app.shared.prompts import sub_agents_content
from app.tools.ragflow_tools import get_assistant_list, create_ask_delete
from app.services.base import A2AAgentService

ragflow_config = sub_agents_content["ragflow"]

service = A2AAgentService(
    name=ragflow_config["name"],
    description=ragflow_config["description"],
    tools=[get_assistant_list, create_ask_delete],
    system_prompt=ragflow_config["system_prompt"],
    port=8003,
)

service.create_agent()
app = service.build_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
```

### 4.5 A2A 工具 (`app/tools/a2a_agent_tools.py`)

三个 LangChain tool，每个封装 HTTP 调用到一个子智能体服务：

```python
import requests
from langchain_core.tools import tool

SUBAGENT_URLS = {
    "network_search": "http://localhost:8001",
    "database_query": "http://localhost:8002",
    "ragflow": "http://localhost:8003",
}

TIMEOUT = 120  # 子智能体执行超时秒数

def _call_subagent(service_key: str, query: str) -> str:
    """通用子智能体调用函数"""
    url = f"{SUBAGENT_URLS[service_key]}/tasks"
    try:
        resp = requests.post(url, json={"query": query}, timeout=TIMEOUT)
        data = resp.json()
        if data.get("status") == "completed":
            return data["result"]
        return f"错误: {data.get('error', '未知错误')}"
    except requests.Timeout:
        return f"错误: 调用超时（{TIMEOUT}秒）"
    except requests.ConnectionError:
        return f"错误: 无法连接到服务 {url}"


@tool
def call_network_search(query: str) -> str:
    """调用网络搜索助手，从互联网检索公开信息。
    适用场景：搜索新闻、政策、行业趋势、网页资料等非内部公开信息。
    """
    return _call_subagent("network_search", query)


@tool
def call_database_query(query: str) -> str:
    """调用数据库查询助手，查询企业内部结构化业务数据。
    适用场景：药品信息、库存、销售记录等数据库业务数据查询。
    """
    return _call_subagent("database_query", query)


@tool
def call_ragflow_query(query: str) -> str:
    """调用RAGFlow助手，查询企业内部私有知识库中的非结构化文档。
    适用场景：PDF、白皮书、研报、制度文件等内部文档的知识检索。
    """
    return _call_subagent("ragflow", query)
```

### 4.6 主智能体 Prompt 重写层 (Query Rewriting Layer)

解耦后子智能体是无状态服务，没有对话上下文。主智能体在调用任何子智能体之前，**必须将用户的模糊自然语言重写为自包含、可直接执行的查询**。

**核心机制**：不是额外写一段代码做重写，而是利用 LLM 在填入 tool 参数时的天然行为——模型根据 system_prompt 的规则，自动把重写后的 query 填入工具参数。

#### 重写规则

| 类别 | 用户原话 | 重写后 query | 目标工具 |
|------|----------|-------------|----------|
| 时间消歧 | "昨天天气如何" | "2026年7月30日 北京 天气 气温 降水 湿度 风速" | call_network_search |
| 时间消歧 | "上个月销售额" | "2026年6月 药品销售总额 按区域汇总" | call_database_query |
| 时间消歧 | "最近一周的新闻" | "2026年7月24日至7月31日 AI行业新闻" | call_network_search |
| 领域补全 | "查一下库存" | "查询所有药品的当前库存量，按仓库和药品名称列出" | call_database_query |
| 术语展开 | "慢病药的销售情况" | "查询高血压、糖尿病、心血管类药品(治疗领域)的销售记录，按区域和金额汇总" | call_database_query |
| 多源分解 | "分析AI对我们业务的影响" | (分两次调用) ① call_network_search("2026年AI技术发展趋势和行业影响") ② call_ragflow_query("公司当前业务线和技术栈，AI可能影响的关键环节") | 多个工具 |
| 角色/指代消解 | "我们的竞品最近有什么动作" | "中国制药行业头部企业 2026年7月 新品发布 市场动态 战略合作" | call_network_search |

#### 重写原则（写入 system_prompt）

1. **时间具体化**：永远把相对时间("昨天/上周/上个月/最近")转为绝对日期或日期范围。当前日期从对话上下文中获取
2. **指代消解**：把"我们公司/竞品/这个/那个"等指代转为具体实体名称
3. **领域补全**：根据用户问题所属领域，补全关键搜索维度。如天气→温度+降水+湿度+风速，药品→名称+规格+库存+效期
4. **自包含**：query 本身是一个完整、独立的问题，子智能体不需要任何额外上下文就能理解并执行
5. **粒度控制**：单次 query 聚焦一个明确目标；复杂任务分解为多次工具调用，每次一个角度
6. **查询语言**：对数据库查询，query 应描述"查什么数据、什么条件、什么聚合维度"，让子智能体的 LLM 自行转换为 SQL；**不要**在 query 中写 SQL 语句

#### 为什么不需要单独的重写模块

```
用户: "昨天天气如何"
  │
  ▼
主智能体 LLM (阅读 system_prompt 中的重写规则)
  │
  │  自动推算：今天=7.31 → 昨天=7.30
  │  自动补全：天气→气温+降水+风速+湿度
  │  填入 tool 参数:
  │    call_network_search(query="2026年7月30日 天气 气温 降水 湿度 风速")
  │
  ▼
HTTP POST /tasks → 网络搜索服务
```

LLM 在 tool calling 时填入参数的过程，就是重写。不需要额外的代码模块。

### 4.7 主智能体改造 (`app/agent/main_agent.py`)

**关键变更：**

```python
# 改前 import
from app.agent.llm import model                     # → 改为 app.shared.llm
from app.agent.prompts import main_agent_content     # → 改为 app.shared.prompts
from app.agent.subagents.database_query_agent import database_query_agent   # → 删除
from app.agent.subagents.knowledge_base_agent import knowledge_base_agent   # → 删除
from app.agent.subagents.network_search_agent import network_search_agent   # → 删除

# 改后 import
from app.shared.llm import model
from app.shared.prompts import main_agent_content
from app.tools.a2a_agent_tools import call_network_search, call_database_query, call_ragflow

# 改前 create_deep_agent
main_agent = create_deep_agent(
    model=model,
    system_prompt=main_agent_content["system_prompt"],
    tools=[generate_markdown, convert_md_to_pdf, read_file_content],
    checkpointer=InMemorySaver(),
    subagents=[database_query_agent, network_search_agent, knowledge_base_agent],
)

# 改后 create_deep_agent
main_agent = create_deep_agent(
    model=model,
    system_prompt=main_agent_content["system_prompt"],
    tools=[
        generate_markdown, convert_md_to_pdf, read_file_content,
        call_network_search, call_database_query, call_ragflow,
    ],
    checkpointer=InMemorySaver(),
)
```

**astream 事件处理变更：**

改前通过检测 `tool_call["name"] == "task"` 来识别子智能体调用，改后子智能体变成普通 tool，不再需要特殊处理：

```python
# 改前
for tool_call in last_msg.tool_calls:
    if tool_call["name"] == "task":
        monitor.report_assistant(tool_call["args"]["subagent_type"], {...})

# 改后 → 直接删除这段特殊处理，或改为检测 A2A 工具名
# monitor 日志由工具内部的 monitor.report_tool() 自动上报
# 但因为子服务运行在独立进程，子服务内没有 WebSocket 推送能力
# 所以主智能体侧的工具调用走 monitor.report_tool()，子服务内部不再推送
```

### 4.8 提示词更新 (`app/prompt/prompts.yml`)

主智能体的 system_prompt 需要从"调度子智能体"改为"调用工具 + Query 重写"模式：

```yaml
main_agent:
  system_prompt: |
    你是金融与电商研究团队负责人，负责理解用户意图、重写查询、调度专家工具、汇总结果。

    【核心职责：Query 重写】
    你调用的每个专家工具都是无状态服务，没有对话上下文，不知道"昨天/上个月/我们公司"
    是什么意思。因此你必须在传入 query 参数之前，把用户的问题重写为自包含的完整查询。

    重写规则：
    1. 时间具体化：相对时间必须转为绝对日期。
       当前日期从对话上下文中推断。例如：
       - "昨天" → 推算为具体日期 "2026年X月X日"
       - "上个月" → "2026年6月"
       - "最近一周" → "2026年7月24日至7月31日"
       - "今年" → "2026年"
    2. 指代消解：代词和简称展开为具体实体。
       - "我们公司" → 根据上下文补全公司名
       - "这个产品" → 具体产品名
       - "竞品" → 明确竞品名称
    3. 领域补全：根据问题领域补全查询维度。
       - 天气 → 气温 + 降水 + 湿度 + 风速 + 天气状况
       - 药品库存 → 药品名称 + 批号 + 数量 + 仓库 + 效期
       - 销售 → 日期 + 数量 + 金额 + 客户 + 区域
       - 行业分析 → 市场规模 + 竞争格局 + 技术趋势 + 政策法规
    4. 自包含：query 必须是完整的独立问题，子智能体仅凭 query 就能理解并执行。
    5. 粒度控制：单次查询聚焦一个明确目标。复杂任务分解为多次工具调用。

    【你掌握的工具】
    专家工具（信息获取）：
      1. call_network_search(query): 互联网公开信息检索（新闻/政策/趋势/天气/百科）
      2. call_database_query(query): 企业结构化数据库查询（药品/库存/销售数据）
      3. call_ragflow_query(query): 企业内部知识库检索（PDF/研报/制度/白皮书）
    文件工具（结果交付）：
      4. generate_markdown(content, filename, path): 生成 Markdown 文档
      5. convert_md_to_pdf(md_filename, pdf_filename): Markdown 转 PDF
      6. read_file_content(filename, instruction): 读取用户上传的附件

    【工作流程】
    第一步：分析用户问题，识别涉及的信息来源
      - 需要公开信息（新闻/政策/百科/天气）→ call_network_search
      - 需要业务数据（药品/库存/销售）→ call_database_query
      - 需要内部文档（研报/制度/白皮书）→ call_ragflow_query
      - 边界不明确 → 多个工具都调用，综合判断
    第二步：对每个要调用的工具，按重写规则构造 query 参数
    第三步：汇总所有专家返回的结果
    第四步：如果用户要求生成文件，调用文件工具；否则直接反馈汇总结果

    【关键约束】
    - 永远先获取信息，再生成文件。禁止在获取信息之前调用文件生成工具
    - query 中不要写 SQL 语句，描述"查什么、什么条件、什么维度"即可
    - 复杂问题先做任务分解(todo-list)，再逐步执行
    - 汇总结果时引用来源，说明信息来自哪个工具

# 子智能体配置保持不变（各自的服务读取各自的配置段）
sub_agents:
  tavily: ...
  db: ...
  ragflow: ...
```

### 4.9 启动脚本 (`start_services.py`)

```python
"""DeepSearch Agents 一键启动脚本

启动 4 个独立 Agent 服务:
  - port 8000: 主智能体
  - port 8001: 网络搜索智能体
  - port 8002: 数据库查询智能体
  - port 8003: RAGFlow 智能体

使用方式: python start_services.py
关闭方式: Ctrl+C
"""

import subprocess
import sys
import signal
import time

SERVICES = [
    ("主智能体", "app.api.server:app", 8000),
    ("网络搜索智能体", "app.services.network_search_service:app", 8001),
    ("数据库查询智能体", "app.services.database_query_service:app", 8002),
    ("RAGFlow智能体", "app.services.ragflow_service:app", 8003),
]

processes = []

def start_service(name, app_path, port):
    cmd = [
        sys.executable, "-m", "uvicorn", app_path,
        "--host", "0.0.0.0",
        "--port", str(port),
    ]
    proc = subprocess.Popen(cmd)
    processes.append((name, proc))
    print(f"  [{name}] 启动中... port={port}, pid={proc.pid}")

def shutdown(sig, frame):
    print("\n正在关闭所有服务...")
    for name, proc in processes:
        print(f"  停止 [{name}] pid={proc.pid}")
        proc.terminate()
    for name, proc in processes:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("所有服务已关闭。")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("=== DeepSearch Agents 服务启动 ===\n")
    for name, app_path, port in SERVICES:
        start_service(name, app_path, port)

    print("\n所有服务已启动，按 Ctrl+C 停止\n")
    print("服务地址:")
    print("  主智能体:         http://localhost:8000")
    print("  网络搜索智能体:    http://localhost:8001")
    print("  数据库查询智能体:  http://localhost:8002")
    print("  RAGFlow智能体:     http://localhost:8003")
    print()

    # 保持主进程运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown(None, None)
```

---

## 五、执行步骤

| 步骤 | 操作 | 产出 |
|------|------|------|
| 1 | 创建 `app/shared/` 目录，移动 `llm.py` 和 `prompts.py` | 共享代码模块 |
| 2 | 创建 `app/services/base.py` A2A 服务基类 | 可复用的服务框架 |
| 3 | 创建 `app/services/network_search_service.py` | port 8001 服务 |
| 4 | 创建 `app/services/database_query_service.py` | port 8002 服务 |
| 5 | 创建 `app/services/ragflow_service.py` | port 8003 服务 |
| 6 | 创建 `app/tools/a2a_agent_tools.py` | 3 个 HTTP 包装工具 |
| 7 | 修改 `app/agent/main_agent.py` | 替换 subagents → A2A tools |
| 8 | 修改 `app/prompt/prompts.yml` | 更新主智能体 prompt |
| 9 | 修改 `app/agent/llm.py` 和 `app/agent/prompts.py` | re-export 保持兼容 |
| 10 | 创建 `start_services.py` | 一键启动脚本 |
| 11 | 端到端测试 | 验证完整链路 |

---

## 六、向后兼容

- `app/agent/llm.py` 保留为 re-export：`from app.shared.llm import *`
- `app/agent/prompts.py` 保留为 re-export：`from app.shared.prompts import *`
- `app/agent/subagents/` 目录保留不删除，旧代码仍可运行
- 如果想回退到旧架构，只需恢复 `main_agent.py` 中的 import 和 `create_deep_agent` 调用

---

## 七、风险与注意事项

1. **子智能体服务启动顺序**：主智能体 (port 8000) 启动时不需要子服务已就绪；A2A 工具在首次调用时才连接子服务，ConnectionError 会被捕获并返回错误消息给模型
2. **超时控制**：A2A 工具设置 120s 超时，防止子服务卡死拖垮主智能体
3. **checkpointer 隔离**：每个服务的 InMemorySaver 完全独立，子服务重启后记忆丢失（后续可换持久化 Backend）
4. **monitor 简化**：子服务不再推送 WebSocket 事件（独立进程无法访问主进程的 WebSocket 连接），可后续通过 A2A streaming 扩展
5. **Windows 兼容**：`start_services.py` 使用纯 Python subprocess，跨平台可用；信号处理在 Windows 上有限制但 `terminate()` 可用
