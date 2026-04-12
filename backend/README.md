# Backend

后端当前默认面向本地可验证链路：

- 数据库：PostgreSQL + pgvector
- 关键词检索：Elasticsearch
- 问答：FastAPI + LangChain
- rerank：可选智谱 `rerank-3`
- 评测：离线 faithfulness

安装：

```powershell
cd backend
D:\anaconda\envs\agent\python.exe -m pip install -e .[dev]
```

如果需要结构化 PDF 抽取：

```powershell
cd backend
D:\anaconda\envs\agent\python.exe -m pip install -e .[pdf-structured]
```

运行：

```powershell
cd backend
D:\anaconda\envs\agent\python.exe -m uvicorn app.main:app --reload
```

测试：

```powershell
cd backend
D:\anaconda\envs\agent\python.exe -m pytest
```

常用 CLI：

```powershell
cd backend
D:\anaconda\envs\agent\python.exe -m app.cli.main import-knowledge --username admin --file data\sample_textbook.txt --book-title "小学数学" --doc-type textbook
D:\anaconda\envs\agent\python.exe -m app.cli.main rebuild-knowledge-index
D:\anaconda\envs\agent\python.exe -m app.cli.main evaluate-faithfulness
```
