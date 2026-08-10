# 历史会话持久化

## 目标

将对话历史存入 MySQL，支持前端浏览和加载历史会话。

## 数据库设计

### 表结构

**sessions** — 会话元数据

| 列 | 类型 | 说明 |
|---|---|---|
| id | INT PK AUTO_INCREMENT | 主键 |
| thread_id | VARCHAR(36) UNIQUE | 对应现有 thread_id |
| title | VARCHAR(200) | 会话标题（从首轮 query 截取或 LLM 生成） |
| status | ENUM('active', 'done') | 会话状态 |
| created_at | DATETIME | 创建时间 |
| updated_at | DATETIME | 最后更新时间 |

**conversations** — 对话轮次

| 列 | 类型 | 说明 |
|---|---|---|
| id | INT PK AUTO_INCREMENT | 主键 |
| session_id | INT FK → sessions.id | 所属会话 |
| turn_index | INT | 轮次序号 |
| user_query | TEXT | 用户问题 |
| assistant_result | TEXT | assistant 回复内容 |
| files | JSON | 输出文件列表 |
| events | JSON | 执行事件流（可选） |
| created_at | DATETIME | 轮次开始时间 |
| finished_at | DATETIME | 轮次结束时间 |

### 写入时机

- sessions：会话创建时 upsert（`INSERT ... ON DUPLICATE KEY UPDATE`）
- conversations：每轮 assistant 回复完成后写入

### 读取 API

- `GET /api/sessions` → 历史会话列表（id, title, 轮次数, 时间）
- `GET /api/sessions/{session_id}/conversations` → 加载完整对话

## 前端改动

- 侧栏新增"历史会话"列表
- 点击可切换加载历史对话
- "新建会话"时创建新 session

## 待讨论

- [ ] events 列是否需要（目前前端已不展示思考过程）
- [ ] 会话标题是否需要 LLM 生成摘要，还是直接截取 query 前 N 字
- [ ] 历史会话是否需要删除功能
