import { Alert, App as AntApp, Button, Popconfirm, Tooltip } from "antd";
import { DeleteOutlined, LogoutOutlined } from "@ant-design/icons";
import { useEffect, useState } from "react";
import { ChatComposer } from "./components/ChatComposer";
import { ConversationThread } from "./components/ConversationThread";
import type { ChatTurn } from "./components/ConversationThread";
import { LoginPage } from "./components/LoginPage";
import { useDeepAgentSession } from "./hooks/useDeepAgentSession";
import { clearUser, getStoredUser, storeUser } from "./lib/auth";
import type { AuthResponse, ConnectionState, UploadedItem } from "./types";

function wsDotColor(state: ConnectionState): string {
  const colors: Record<ConnectionState, string> = {
    connected: "#10b981",
    connecting: "#f59e0b",
    reconnecting: "#f59e0b",
    closed: "#ef4444",
  };
  return colors[state];
}

function wsDotLabel(state: ConnectionState): string {
  const labels: Record<ConnectionState, string> = {
    connected: "WebSocket 已连接",
    connecting: "WebSocket 连接中",
    reconnecting: "WebSocket 重连中",
    closed: "WebSocket 已关闭",
  };
  return labels[state];
}

function createTurn(content: string): ChatTurn {
  return {
    id: crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}`,
    content,
    events: [],
    files: [],
    isRunning: true,
    result: "",
    timestamp: new Date().toISOString()
  };
}

export default function App() {
  const { message } = AntApp.useApp();
  const [user, setUser] = useState<AuthResponse | null>(getStoredUser);
  const [query, setQuery] = useState("");
  const [stagedItems, setStagedItems] = useState<UploadedItem[]>([]);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const session = useDeepAgentSession(user?.user_id ?? null);

  useEffect(() => {
    setTurns((previous) => {
      if (previous.length === 0) {
        return previous;
      }

      const latestTurn = previous[previous.length - 1];
      // 没有活跃任务也没有新结果时，保留已加载的历史轮次不动
      if (!latestTurn.isRunning && !session.isRunning && !session.result) {
        return previous;
      }

      const nextLatestTurn = {
        ...latestTurn,
        events: session.events,
        files: session.files.length > 0 ? session.files : latestTurn.files,
        isRunning: session.isRunning,
        result: session.result || latestTurn.result,
      };

      return [...previous.slice(0, -1), nextLatestTurn];
    });
  }, [session.events, session.files, session.isRunning, session.result]);

  async function handleSubmit() {
    const cleanQuery = query.trim();
    if (!cleanQuery) {
      message.warning("请输入研搜任务");
      return;
    }

    const nextTurn = createTurn(cleanQuery);
    setTurns((previous) => [...previous, nextTurn]);
    setQuery("");

    try {
      await session.submitTask(cleanQuery);
      message.success("任务已启动，执行过程会显示在对话中");
    } catch (error) {
      setTurns((previous) =>
        previous.map((turn) =>
          turn.id === nextTurn.id
            ? {
                ...turn,
                isRunning: false,
                result: error instanceof Error ? error.message : "任务启动失败"
              }
            : turn
        )
      );
      message.error(error instanceof Error ? error.message : "任务启动失败");
    }
  }

  async function handleCancel() {
    try {
      const response = await session.cancelCurrentTask();
      message.info(response.status === "cancelling" ? "取消请求已发送，正在等待当前调用结束" : "任务已取消");
    } catch (error) {
      message.error(error instanceof Error ? error.message : "取消任务失败");
    }
  }

  async function handleUpload(items: UploadedItem[]) {
    try {
      const response = await session.uploadFiles(items);
      setStagedItems([]);
      message.success(`已上传 ${response.files.length} 个文件`);
    } catch (error) {
      message.error(error instanceof Error ? error.message : "上传失败");
    }
  }

  function handleNewSession() {
    session.resetSession();
    setTurns([]);
    setQuery("");
    setStagedItems([]);
    session.loadSessionsList();
  }

  function handleLogin(nextUser: AuthResponse) {
    storeUser(nextUser);
    setUser(nextUser);
    // 切换用户后生成全新会话，避免沿用上一个用户在 localStorage 中的 thread_id
    session.resetSession();
    setTurns([]);
    setQuery("");
    setStagedItems([]);
  }

  function handleLogout() {
    clearUser();
    setUser(null);
    session.resetSession();
    setTurns([]);
    setQuery("");
    setStagedItems([]);
  }

  async function handleLoadSession(threadId: string) {
    try {
      const turns = await session.loadSession(threadId);
      setTurns(turns);
      setQuery("");
      setStagedItems([]);
    } catch {
      message.error("加载历史会话失败");
    }
  }

  async function handleDeleteSession(threadId: string) {
    try {
      await session.removeSession(threadId);
      // 如果删除的是当前活跃会话，清空界面
      if (threadId === session.threadId) {
        session.resetSession();
        setTurns([]);
        setQuery("");
        setStagedItems([]);
      }
      message.success("已删除会话");
    } catch {
      message.error("删除会话失败");
    }
  }

  const dotColor = wsDotColor(session.connectionState);
  const dotLabel = wsDotLabel(session.connectionState);

  if (!user) {
    return <LoginPage onLogin={handleLogin} />;
  }

  return (
    <div className="chat-app-shell min-h-dvh">
      <aside className="chat-sidebar" aria-label="会话信息">
        <div className="sidebar-brand">
          <span className="panel-kicker sidebar-brand-kicker">DEEPSEARCH</span>
        </div>

        <Button className="new-chat-button" block onClick={handleNewSession}>
          新建会话
        </Button>

        <div className="sidebar-section">
          <div className="sidebar-section-header">历史会话</div>
          <div className="session-list">
            {session.sessions.length === 0 ? (
              <div className="session-list-empty">暂无历史会话</div>
            ) : (
              session.sessions.map((s) => {
                const isActive = s.thread_id === session.threadId;
                return (
                  <div
                    className={`session-item${isActive ? " session-item--active" : ""}`}
                    key={s.thread_id}
                    onClick={() => {
                      if (!isActive) {
                        handleLoadSession(s.thread_id);
                      }
                    }}
                  >
                    <div className="session-item-title">{s.title}</div>
                    <div className="session-item-meta">
                      <span>{s.turn_count} 轮</span>
                      <span>{s.updated_at?.slice(0, 10)}</span>
                    </div>
                    <Popconfirm
                      cancelText="取消"
                      okText="删除"
                      placement="right"
                      title="确认删除该会话？"
                      onConfirm={(e) => {
                        e?.stopPropagation();
                        handleDeleteSession(s.thread_id);
                      }}
                      onCancel={(e) => e?.stopPropagation()}
                    >
                      <Button
                        className="session-item-delete"
                        danger
                        icon={<DeleteOutlined />}
                        size="small"
                        type="text"
                        onClick={(e) => e.stopPropagation()}
                      />
                    </Popconfirm>
                  </div>
                );
              })
            )}
          </div>
        </div>

        <div className="sidebar-footer">
          <div className="sidebar-footer-user">
            <span className="sidebar-avatar">{user.username.charAt(0).toUpperCase()}</span>
            <div className="sidebar-footer-meta">
              <span className="sidebar-footer-name">{user.username}</span>
              <span className="sidebar-footer-label">当前用户</span>
            </div>
          </div>
          <Tooltip title="退出登录" placement="top">
            <Button
              className="sidebar-logout"
              aria-label="退出登录"
              icon={<LogoutOutlined />}
              size="small"
              type="text"
              onClick={handleLogout}
            />
          </Tooltip>
        </div>
      </aside>

      <main className="chat-main">
        <Tooltip title={dotLabel}>
          <span
            aria-label={dotLabel}
            className="ws-status-dot"
            style={{ background: dotColor }}
          />
        </Tooltip>

        {session.lastError ? (
          <Alert
            className="chat-alert"
            message={session.lastError}
            showIcon
            type="error"
          />
        ) : null}

        <section className="chat-stream-panel">
          <ConversationThread
            onUseExample={setQuery}
            turns={turns}
            username={user.username}
          />
        </section>

        <ChatComposer
          isCancelling={session.isCancelling}
          isRunning={session.isRunning}
          isUploading={session.isUploading}
          onCancel={handleCancel}
          onNewSession={handleNewSession}
          onQueryChange={setQuery}
          onStagedItemsChange={setStagedItems}
          onSubmit={handleSubmit}
          onUpload={handleUpload}
          query={query}
          stagedItems={stagedItems}
          uploadedItems={session.uploadedItems}
        />
      </main>
    </div>
  );
}
