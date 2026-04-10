# 出版社 AI 助手

一个基于 LangChain 的出版社场景 AI 助手项目，当前支持登录鉴权、知识导入、RAG 问答、拍照搜题、历史记录和管理后台式前端。

当前本地已验证的模型提供方：

- OpenAI
- ARK / 豆包（OpenAI 兼容）
- GLM / 智谱（OpenAI 兼容）

当前实际跑通的配置为：`GLM + glm-4.5-air + glm-4.5v + embedding-3`。

## 项目结构

```text
backend/    FastAPI + LangChain + SQLAlchemy + FAISS
frontend/   React + Vite
docs/       技术文档
scripts/    Windows 启动脚本
```

## 环境要求

- Python 3.11+
- Node.js 20+
- 建议使用已验证环境：`D:\anaconda\envs\agent`

## 关键配置

配置文件位于根目录 [.env](E:\daima\agent\ai_assistant\.env)。

当前已支持三套模型配置：

### OpenAI

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
OPENAI_VISION_MODEL=
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### ARK / 豆包

```env
LLM_PROVIDER=ark
ARK_API_KEY=
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_CHAT_MODEL=
ARK_VISION_MODEL=
ARK_EMBEDDING_MODEL=
```

### GLM / 智谱

```env
LLM_PROVIDER=glm
GLM_API_KEY=
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
GLM_CHAT_MODEL=glm-4.5-air
GLM_VISION_MODEL=glm-4.5v
GLM_EMBEDDING_MODEL=embedding-3
```

说明：

- 文本问答使用 `CHAT_MODEL`
- 拍照搜题优先使用 `VISION_MODEL`
- 如果未配置视觉模型，会自动回退到文本模型
- RAG 向量化使用 `EMBEDDING_MODEL`

## 快速启动

### 方式一：使用脚本

启动后端：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_backend.ps1
```

开发模式启动后端：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_backend.ps1 -Reload
```

启动前端：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_frontend.ps1
```

分别弹出两个窗口启动前后端：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_all.ps1
```

### 方式二：手动启动

后端：

```powershell
cd E:\daima\agent\ai_assistant\backend
D:\anaconda\envs\agent\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

前端：

```powershell
cd E:\daima\agent\ai_assistant\frontend
npm.cmd run dev -- --host 127.0.0.1 --port 5173
```

## 访问地址

- 前端：`http://127.0.0.1:5173`
- 后端：`http://127.0.0.1:8000`
- 健康检查：`http://127.0.0.1:8000/health`
- Swagger：`http://127.0.0.1:8000/docs`

## 默认账号

- 用户名：`admin`
- 密码：`admin123456`

## 已验证能力

- 登录
- 管理员导入知识文档
- FAISS 检索
- 文本问答
- 拍照搜题 OCR 链路
- 对话历史记录

## 测试

运行后端测试：

```powershell
cd E:\daima\agent\ai_assistant\backend
D:\anaconda\envs\agent\python.exe -m pytest
```

当前已验证：`5 passed`

## 文档导航

查看 [docs/README.md](E:\daima\agent\ai_assistant\docs\README.md)。

## 当前注意事项

- `.env` 中的 `JWT_SECRET_KEY` 仍是开发值，建议改成更长的随机字符串
- FastAPI 当前仍使用 `on_event`，后续可迁移到 lifespan
- Windows 终端直接打印中文时可能受本地编码影响，但不影响浏览器和 API 实际返回内容# agent_ai_publisher
