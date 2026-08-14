export type ConnectionState = "connecting" | "connected" | "reconnecting" | "closed";

export interface AuthResponse {
  user_id: number;
  username: string;
}

export type MonitorEventName =
  | "session_created"
  | "tool_start"
  | "assistant_call"
  | "task_result"
  | "task_cancelled"
  | "error"
  | string;

export interface MonitorMessage {
  type: "monitor_event";
  event: MonitorEventName;
  message: string;
  data: Record<string, unknown>;
  timestamp: string;
}

export interface PongMessage {
  type: "pong";
  message: string;
}

export type SocketMessage = MonitorMessage | PongMessage;

export interface TaskResponse {
  status: "started" | string;
  thread_id: string;
}

export interface CancelTaskResponse {
  status: "cancelled" | "cancelling" | string;
  thread_id: string;
  message?: string;
}

export interface UploadResponse {
  status: "uploaded" | string;
  files: string[];
}

export interface OutputFile {
  name: string;
  type: "file" | string;
  path: string;
  size: number;
  mtime: number;
}

export interface FileListResponse {
  files?: OutputFile[];
  error?: string;
}

export interface UploadedItem {
  uid: string;
  name: string;
  size: number;
  raw: File;
}

export interface SessionSummary {
  id: number;
  thread_id: string;
  title: string;
  status: string;
  turn_count: number;
  updated_at: string;
}

export interface ConversationTurn {
  turn_index: number;
  user_query: string;
  assistant_result: string;
  files: OutputFile[];
  created_at: string;
}

export interface SessionsResponse {
  sessions: SessionSummary[];
}

export interface ConversationsResponse {
  thread_id: string;
  turns: ConversationTurn[];
}

export interface DeleteSessionResponse {
  status: string;
  thread_id: string;
}
