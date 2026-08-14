import type { AuthResponse } from "../types";

const AUTH_KEY = "deepsearch.user";

export function getStoredUser(): AuthResponse | null {
  try {
    const raw = window.localStorage.getItem(AUTH_KEY);
    return raw ? (JSON.parse(raw) as AuthResponse) : null;
  } catch {
    return null;
  }
}

export function storeUser(user: AuthResponse): void {
  window.localStorage.setItem(AUTH_KEY, JSON.stringify(user));
}

export function clearUser(): void {
  window.localStorage.removeItem(AUTH_KEY);
}
