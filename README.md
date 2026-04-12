# 出版社 AI 助手

面向出版社、教材和教辅运营场景的 AI 助手项目。当前版本重点是把本地可验证的检索链路先跑通：

- 基础设施：`docker-compose` 一键启动 `PostgreSQL + pgvector + Elasticsearch`
- 知识检索：书籍重新导入到 PostgreSQL 和 Elasticsearch，放弃旧的 SQLite + FAISS 向量索引
- 问答链路：FastAPI + LangChain + 混合检索 + 可选智谱 `rerank-3`
- 评测：离线 faithfulness 评测，读取评测集、调用 judge 模型、输出分数和报告

## 项目结构

```text
backend/    FastAPI + SQLAlchemy + PostgreSQL/pgvector + Elasticsearch
frontend/   React + Vite
docs/       设计与接口文档
scripts/    Windows 启动脚本
docker-compose.yml
```

## 快速开始

1. 复制 `.env.example` 为 `.env`
2. 启动基础设施

```powershell
docker compose up -d
```

3. 安装后端依赖

```powershell
cd backend
D:\anaconda\envs\agent\python.exe -m pip install -e .[dev]
```

如果你要启用结构化 PDF 提取，再安装可选依赖：

```powershell
cd backend
D:\anaconda\envs\agent\python.exe -m pip install -e .[pdf-structured]
```

4. 启动后端

```powershell
cd backend
D:\anaconda\envs\agent\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

5. 启动前端

```powershell
cd frontend
npm.cmd run dev -- --host 127.0.0.1 --port 5173
```

## 关键配置

`.env.example` 已包含完整字段，重点关注：

- `DATABASE_URL=postgresql+psycopg://ai_assistant:ai_assistant@127.0.0.1:5432/ai_assistant`
- `ELASTICSEARCH_URL=http://127.0.0.1:9200`
- `LLM_PROVIDER`
- 对应 provider 的 `CHAT_MODEL` / `EMBEDDING_MODEL` / `VISION_MODEL`
- `ENABLE_RERANK=true`
- `ZHIPU_RERANK_API_KEY=`
- `FAITHFULNESS_DATASET_PATH`
- `FAITHFULNESS_REPORT_PATH`

说明：

- 未配置 `ZHIPU_RERANK_API_KEY` 时会自动关闭外部 rerank，保留混合检索结果。
- `PyMuPDF4LLM` 是可选依赖，不再作为唯一 PDF 方案；默认仍可退回 `pypdf`。
- conversations 默认直接写当前 `DATABASE_URL`。旧 SQLite 会话如需迁移，可用 CLI 做一次性导入。

## 常用命令

导入书籍：

```powershell
cd backend
D:\anaconda\envs\agent\python.exe -m app.cli.main import-knowledge --username admin --file data\sample_textbook.txt --book-title "小学数学" --doc-type textbook
```

重建书籍索引：

```powershell
cd backend
D:\anaconda\envs\agent\python.exe -m app.cli.main rebuild-knowledge-index
```

离线评测 faithfulness：

```powershell
cd backend
D:\anaconda\envs\agent\python.exe -m app.cli.main evaluate-faithfulness
```

迁移旧 SQLite conversations：

```powershell
cd backend
D:\anaconda\envs\agent\python.exe -m app.cli.main migrate-conversations --source-sqlite-path ..\data\app.db
```

## API

- 前端：`http://127.0.0.1:5173`
- 后端：`http://127.0.0.1:8000`
- 健康检查：`http://127.0.0.1:8000/health`
- Swagger：`http://127.0.0.1:8000/docs`

默认管理员：

- 用户名：`admin`
- 密码：`admin123456`

## 测试

```powershell
cd backend
D:\anaconda\envs\agent\python.exe -m pytest
```
