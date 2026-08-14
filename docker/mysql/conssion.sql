-- 用户与对话持久化（重新设计版）
-- 包含三张表，按依赖顺序创建：users → sessions → conversations
--
-- 使用说明：在 deepsearch_db 库中手动执行本文件一次即可：
--   mysql -uroot -proot deepsearch_db < conssion.sql
-- 或进入 mysql 客户端后：source /path/to/conssion.sql
--
-- 注意：本文件未被 docker-entrypoint-initdb.d 自动加载（compose 只挂了 mysql.sql），
-- 重建容器后需手动重新执行。

USE deepsearch_db;

-- ---------------------------------------------------------------------------
-- 1. 用户表（登录系统 + 长期记忆/会话按用户隔离）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id            INT PRIMARY KEY AUTO_INCREMENT,
    username      VARCHAR(50)  NOT NULL,
    password_hash VARCHAR(256) NOT NULL,
    created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_users_username (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ---------------------------------------------------------------------------
-- 2. 会话表
--    user_id 为 NULL 表示未登录/游客会话（前端登录为必选项，正常会话都会有 user_id）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    id              INT PRIMARY KEY AUTO_INCREMENT,
    thread_id       VARCHAR(36) NOT NULL,
    user_id         INT         NULL,
    title           VARCHAR(200) NOT NULL DEFAULT '',
    status          ENUM('active', 'done') NOT NULL DEFAULT 'active',
    context_summary TEXT        NULL,
    turn_count      INT         NOT NULL DEFAULT 0,
    created_at      DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_sessions_thread_id (thread_id),
    KEY idx_sessions_user_id (user_id),
    CONSTRAINT fk_sessions_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ---------------------------------------------------------------------------
-- 3. 对话轮次表
--    (session_id, turn_index) 唯一，防止并发重复写入同一轮
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS conversations (
    id               INT PRIMARY KEY AUTO_INCREMENT,
    session_id       INT  NOT NULL,
    turn_index       INT  NOT NULL,
    user_query       TEXT NOT NULL,
    assistant_result TEXT NULL,
    summary          TEXT NULL,
    files            JSON NULL,
    created_at       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at      DATETIME NULL,
    UNIQUE KEY uk_conversations_session_turn (session_id, turn_index),
    CONSTRAINT fk_conversations_session FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
