import { API_BASE_URL } from "./config";
import type {
  AuthResponse,
  CancelTaskResponse,
  ConversationsResponse,
  DeleteSessionResponse,
  FileListResponse,
  SessionsResponse,
  TaskResponse,
  UploadResponse,
} from "../types";

function apiUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

async function requestJson<T>(input: RequestInfo | URL, init?: RequestInit): Promise<T> {
  const response = await fetch(input, init);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const message =
      typeof payload === "object" && payload && "detail" in payload
        ? String(payload.detail)
        : `HTTP ${response.status}`;
    throw new Error(message);
  }

  return payload as T;
}

export async function startTask(
  query: string,
  threadId: string,
  userId?: number | null
): Promise<TaskResponse> {
  return requestJson<TaskResponse>(apiUrl("/api/task"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      query,
      thread_id: threadId,
      user_id: userId ?? null
    })
  });
}

export async function login(username: string, password: string): Promise<AuthResponse> {
  return requestJson<AuthResponse>(apiUrl("/api/auth/login"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password })
  });
}

export async function register(username: string, password: string): Promise<AuthResponse> {
  return requestJson<AuthResponse>(apiUrl("/api/auth/register"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password })
  });
}

export async function cancelTask(threadId: string): Promise<CancelTaskResponse> {
  return requestJson<CancelTaskResponse>(apiUrl(`/api/task/${encodeURIComponent(threadId)}/cancel`), {
    method: "POST"
  });
}

export async function uploadSessionFiles(
  files: File[],
  threadId: string
): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("thread_id", threadId);
  files.forEach((file) => formData.append("files", file));

  return requestJson<UploadResponse>(apiUrl("/api/upload"), {
    method: "POST",
    body: formData
  });
}

export async function listSessionFiles(path: string): Promise<FileListResponse> {
  const url = new URL(apiUrl("/api/files"));
  url.searchParams.set("path", path);
  return requestJson<FileListResponse>(url);
}

export function getDownloadUrl(path: string): string {
  const url = new URL(apiUrl("/api/download"));
  url.searchParams.set("path", path);
  return url.toString();
}

export async function listSessions(userId?: number | null): Promise<SessionsResponse> {
  const url = new URL(apiUrl("/api/sessions"));
  if (userId != null) {
    url.searchParams.set("user_id", String(userId));
  }
  return requestJson<SessionsResponse>(url);
}

export async function getSessionConversations(
  threadId: string
): Promise<ConversationsResponse> {
  return requestJson<ConversationsResponse>(
    apiUrl(`/api/sessions/${encodeURIComponent(threadId)}/conversations`)
  );
}

export async function deleteSession(
  threadId: string
): Promise<DeleteSessionResponse> {
  return requestJson<DeleteSessionResponse>(
    apiUrl(`/api/sessions/${encodeURIComponent(threadId)}`),
    { method: "DELETE" }
  );
}
