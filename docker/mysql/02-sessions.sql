-- 对话历史持久化
-- sessions: 会话元数据
-- conversations: 每轮对话内容
--
-- 使用说明：在 deepsearch_db 库中手动执行本文件，或在 docker compose down -v 后重建容器自动加载

USE deepsearch_db;

CREATE TABLE IF NOT EXISTS sessions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    thread_id VARCHAR(36) NOT NULL UNIQUE,
    title VARCHAR(200) NOT NULL,
    status ENUM('active', 'done') DEFAULT 'active',
    context_summary TEXT,
    turn_count INT DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS conversations (
    id INT PRIMARY KEY AUTO_INCREMENT,
    session_id INT NOT NULL,
    turn_index INT NOT NULL,
    user_query TEXT NOT NULL,
    assistant_result TEXT,
    summary TEXT,
    files JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    finished_at DATETIME,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
