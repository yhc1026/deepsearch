import { LockOutlined, UserOutlined } from "@ant-design/icons";
import { Button, Form, Input, Typography, message } from "antd";
import { useState } from "react";
import { login, register } from "../lib/api";
import type { AuthResponse } from "../types";

interface Props {
  onLogin: (user: AuthResponse) => void;
}

interface FormValues {
  username: string;
  password: string;
}

export function LoginPage({ onLogin }: Props) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(values: FormValues) {
    setLoading(true);
    try {
      const user =
        mode === "login"
          ? await login(values.username, values.password)
          : await register(values.username, values.password);
      onLogin(user);
    } catch (error) {
      message.error(error instanceof Error ? error.message : "操作失败");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="login-shell">
      <div className="login-card">
        <div className="login-brand">DEEPSEARCH</div>
        <Typography.Title level={3} className="login-title">
          {mode === "login" ? "登录" : "注册"}
        </Typography.Title>
        <Typography.Paragraph type="secondary" className="login-subtitle">
          登录后才能使用 DeepSearch 深度研搜助手
        </Typography.Paragraph>

        <Form layout="vertical" onFinish={handleSubmit} requiredMark={false}>
          <Form.Item
            name="username"
            label="用户名"
            rules={[{ required: true, message: "请输入用户名" }]}
          >
            <Input size="large" prefix={<UserOutlined />} placeholder="用户名" autoFocus />
          </Form.Item>
          <Form.Item
            name="password"
            label="密码"
            rules={[{ required: true, message: "请输入密码" }]}
          >
            <Input.Password size="large" prefix={<LockOutlined />} placeholder="密码" />
          </Form.Item>
          <Button type="primary" htmlType="submit" block size="large" loading={loading}>
            {mode === "login" ? "登录" : "注册"}
          </Button>
        </Form>

        <div className="login-switch">
          {mode === "login" ? "还没有账号？" : "已有账号？"}
          <a onClick={() => setMode(mode === "login" ? "register" : "login")}>
            {mode === "login" ? "去注册" : "去登录"}
          </a>
        </div>
      </div>
    </div>
  );
}
