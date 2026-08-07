import "antd/dist/reset.css";
import { App as AntApp, ConfigProvider } from "antd";
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: "#3b82f6",
          colorSuccess: "#10b981",
          colorWarning: "#f59e0b",
          colorError: "#ef4444",
          colorInfo: "#3b82f6",
          colorBgBase: "#ffffff",
          colorBgContainer: "#ffffff",
          colorBorder: "#e2e8f0",
          colorText: "#1e293b",
          colorTextSecondary: "#64748b",
          borderRadius: 8,
          fontFamily:
            "'Inter', 'PingFang SC', 'Microsoft YaHei', system-ui, -apple-system, sans-serif",
          fontFamilyCode:
            "'JetBrains Mono', 'SFMono-Regular', Consolas, 'Liberation Mono', monospace",
        },
        components: {
          Button: {
            controlHeightLG: 44,
            borderRadius: 8,
            primaryShadow: "0 1px 3px rgba(59, 130, 246, 0.12)",
          },
          Input: {
            activeBorderColor: "#3b82f6",
            hoverBorderColor: "#93c5fd",
            borderRadius: 8,
          },
        },
      }}
    >
      <AntApp>
        <App />
      </AntApp>
    </ConfigProvider>
  </React.StrictMode>
);
