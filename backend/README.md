# 出版社 AI 助手

基于 LangChain 的出版社场景 AI 助手单仓库项目。V1 聚焦编辑 / 教辅运营场景，提供登录鉴权、知识导入、RAG 问答、历史查询，以及轻量 Web 门户。

## 项目简介

这个项目面向出版社、教辅和内容运营场景，目标是搭建一个可运行、可理解、可迭代的 Agent 系统，而不是一次性的 Demo。当前版本优先解决三件事：

- 让管理员能导入图书或教辅资料
- 让登录用户基于知识库进行问答
- 让整个系统具备后续扩展到多模态、多 Agent、更多出版工具的工程基础

## 技术选型

- 后端：Python、FastAPI、LangChain、SQLAlchemy、FAISS、Typer
- 前端：React、Vite、React Router
- 鉴权：JWT
- 知识库：本地文件导入 + OpenAI Embeddings + FAISS
- 可观测性：结构化日志 + LangSmith 可选配置

## 当前能力

- 用户登录：账号密码 + JWT
- 角色：管理员 / 普通用户
- 文档导入：TXT、Markdown、PDF、DOCX
- RAG：分层切片、向量检索、按角色过滤、按书名 / 文档类型过滤
- Agent：知识库检索工具 + 计算器工具
- 问答结果：返回答案、是否有依据、引用来源
- 历史记录：保存和查询，不自动参与后续问答
- 入口：CLI + HTTP API + React 前端

## 目录结构

```text
backend/
  app/
  tests/
  data/
frontend/
.env.example
README.md
```

## 环境要求

- Python 3.11+
- Node.js 20+
- OpenAI API Key

## 安装步骤

### 1. 配置环境变量

复制根目录 `.env.example` 为 `.env`，并至少设置：

- `OPENAI_API_KEY`
- `JWT_SECRET_KEY`
- `DEFAULT_ADMIN_USERNAME`
- `DEFAULT_ADMIN_PASSWORD`

### 2. 安装后端依赖

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

### 3. 安装前端依赖

```bash
cd frontend
npm install
```

## 环境变量说明

- `OPENAI_API_KEY`: OpenAI 密钥
- `OPENAI_MODEL`: 对话模型，默认 `gpt-4o-mini`
- `OPENAI_EMBEDDING_MODEL`: 向量模型，默认 `text-embedding-3-small`
- `DATABASE_URL`: SQLite 或未来的 PostgreSQL 连接串
- `JWT_SECRET_KEY`: JWT 密钥
- `DEFAULT_ADMIN_USERNAME`: 默认管理员用户名
- `DEFAULT_ADMIN_PASSWORD`: 默认管理员密码
- `LANGSMITH_API_KEY`: 可选，LangSmith 追踪密钥
- `LANGSMITH_TRACING`: 可选，是否启用 LangSmith
- `UPLOAD_DIR`: 上传文件目录
- `VECTOR_STORE_DIR`: 向量索引目录

## 启动方式

### 启动后端 API

```bash
cd backend
uvicorn app.main:app --reload
```

### 初始化默认管理员

后端启动时会自动确保 `.env` 中的默认管理员存在。也可以手动执行：

```bash
cd backend
python -m app.cli.main bootstrap-admin
```

### 启动前端

```bash
cd frontend
npm run dev
```

## CLI 示例

```bash
cd backend
python -m app.cli.main create-admin --username editor_admin --password strong-password
python -m app.cli.main import-knowledge --username admin --file data/sample_textbook.txt --book-title "小学数学" --doc-type textbook
python -m app.cli.main ask --username admin --role admin --question "这本教辅适合几年级？"
```

## API 示例

### 登录

```bash
curl -X POST http://127.0.0.1:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123456"}'
```

### 导入知识

```bash
curl -X POST http://127.0.0.1:8000/api/knowledge/import \
  -H "Authorization: Bearer <token>" \
  -F "file=@backend/data/sample_textbook.txt" \
  -F "book_title=小学数学" \
  -F "doc_type=textbook" \
  -F "allowed_role=user"
```

### 问答

```bash
curl -X POST http://127.0.0.1:8000/api/chat/ask \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"question": "这本教辅适合几年级？", "book_title": "小学数学"}'
```

## 测试

```bash
cd backend
pytest
```

## 后续扩展建议

- 接入 OCR 与语音识别，扩展到拍照搜题和语音问答
- 为检索增加 rerank 和更细粒度元数据过滤
- 引入 PostgreSQL、对象存储和云向量库
- 将管理员 / 普通用户扩展为通用 RBAC
- 在 `app/agents/orchestration` 中接入多 Agent 编排

## 当前限制

- V1 仅实现 OpenAI 提供方
- 对话历史只保存和查询，不自动作为长期记忆注入
- PDF / DOCX 解析依赖原始文档质量
- 当前默认本地运行，未包含生产部署脚本
