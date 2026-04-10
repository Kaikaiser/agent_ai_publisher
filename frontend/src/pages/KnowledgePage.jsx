import { useEffect, useState } from 'react'

import { apiRequest } from '../api/client'

function buildDocumentQuery(filters) {
  const params = new URLSearchParams()
  if (filters.query.trim()) params.set('query', filters.query.trim())
  if (filters.bookTitle.trim()) params.set('book_title', filters.bookTitle.trim())
  if (filters.docType.trim()) params.set('doc_type', filters.docType.trim())
  const query = params.toString()
  return query ? `/knowledge/documents?${query}` : '/knowledge/documents'
}

export default function KnowledgePage() {
  const [file, setFile] = useState(null)
  const [bookTitle, setBookTitle] = useState('')
  const [docType, setDocType] = useState('textbook')
  const [allowedRole, setAllowedRole] = useState('user')
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [documents, setDocuments] = useState([])
  const [loading, setLoading] = useState(false)
  const [filters, setFilters] = useState({ query: '', bookTitle: '', docType: '' })
  const [reindexResult, setReindexResult] = useState(null)

  const loadDocuments = async (nextFilters = filters) => {
    setLoading(true)
    setError('')
    try {
      const data = await apiRequest(buildDocumentQuery(nextFilters))
      setDocuments(data.items)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadDocuments()
  }, [])

  const handleSubmit = async (event) => {
    event.preventDefault()
    setError('')
    setMessage('')
    setReindexResult(null)
    const formData = new FormData()
    formData.append('file', file)
    formData.append('book_title', bookTitle)
    formData.append('doc_type', docType)
    formData.append('allowed_role', allowedRole)
    try {
      const data = await apiRequest('/knowledge/import', { method: 'POST', body: formData })
      setMessage(`导入成功：${data.message}`)
      setFile(null)
      await loadDocuments()
    } catch (err) {
      setError(err.message)
    }
  }

  const handleFilterSubmit = async (event) => {
    event.preventDefault()
    await loadDocuments(filters)
  }

  const handleDelete = async (documentId) => {
    setError('')
    setMessage('')
    if (!window.confirm('确认删除这份知识文档并重建索引吗？')) {
      return
    }
    try {
      const data = await apiRequest(`/knowledge/documents/${documentId}`, { method: 'DELETE' })
      setMessage(`已删除 ${data.filename}，并完成索引重建。`)
      setReindexResult(data)
      await loadDocuments()
    } catch (err) {
      setError(err.message)
    }
  }

  const handleReindex = async () => {
    setError('')
    setMessage('')
    try {
      const data = await apiRequest('/knowledge/reindex', { method: 'POST' })
      setReindexResult(data)
      setMessage(`重建完成：索引 ${data.documents_indexed} 份文档，生成 ${data.chunks_indexed} 个切片。`)
      await loadDocuments()
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <section className="page-section">
      <div className="page-intro">
        <h2>知识库管理</h2>
        <p>导入教材与资料后，可以在这里筛选、巡检、删除和重建本地索引。</p>
      </div>

      <div className="knowledge-layout">
        <div className="panel">
          <div className="panel-toolbar">
            <div>
              <h3>导入知识文档</h3>
              <p className="muted">支持 TXT、Markdown、PDF、DOCX，导入后立即切片并写入向量索引。</p>
            </div>
            <button className="secondary-button" type="button" onClick={handleReindex}>
              重建全部索引
            </button>
          </div>

          <form onSubmit={handleSubmit} className="stack">
            <input type="file" onChange={(e) => setFile(e.target.files?.[0] || null)} />
            <div className="grid-two">
              <input value={bookTitle} onChange={(e) => setBookTitle(e.target.value)} placeholder="书名，如：小学数学" />
              <input value={docType} onChange={(e) => setDocType(e.target.value)} placeholder="文档类型，如：textbook" />
            </div>
            <select value={allowedRole} onChange={(e) => setAllowedRole(e.target.value)}>
              <option value="user">管理员和普通用户均可访问</option>
              <option value="admin">仅管理员可访问</option>
            </select>
            <button className="primary-button" type="submit" disabled={!file}>导入知识文档</button>
          </form>

          {message && <p className="success-text top-gap">{message}</p>}
          {error && <p className="error-text top-gap">{error}</p>}
          {reindexResult && (
            <div className="top-gap subtle-panel">
              <strong>最近一次索引结果</strong>
              <p className="muted">
                已索引 {reindexResult.documents_indexed} 份文档，生成 {reindexResult.chunks_indexed} 个切片。
              </p>
              {reindexResult.missing_files?.length > 0 && (
                <p className="error-text">缺失文件：{reindexResult.missing_files.join('，')}</p>
              )}
            </div>
          )}
        </div>

        <div className="panel">
          <div className="panel-toolbar">
            <div>
              <h3>已导入文档</h3>
              <p className="muted">当前共 {documents.length} 份文档。</p>
            </div>
          </div>

          <form onSubmit={handleFilterSubmit} className="stack">
            <div className="grid-three">
              <input
                value={filters.query}
                onChange={(e) => setFilters((current) => ({ ...current, query: e.target.value }))}
                placeholder="搜索文件名、书名、类型或导入人"
              />
              <input
                value={filters.bookTitle}
                onChange={(e) => setFilters((current) => ({ ...current, bookTitle: e.target.value }))}
                placeholder="按书名筛选"
              />
              <input
                value={filters.docType}
                onChange={(e) => setFilters((current) => ({ ...current, docType: e.target.value }))}
                placeholder="按文档类型筛选"
              />
            </div>
            <div className="button-row">
              <button className="primary-button" type="submit">刷新列表</button>
              <button
                className="ghost-button"
                type="button"
                onClick={() => {
                  const nextFilters = { query: '', bookTitle: '', docType: '' }
                  setFilters(nextFilters)
                  loadDocuments(nextFilters)
                }}
              >
                清空筛选
              </button>
            </div>
          </form>

          {loading ? (
            <div className="empty-box muted">正在加载文档列表...</div>
          ) : documents.length > 0 ? (
            <div className="knowledge-list">
              {documents.map((item) => (
                <article key={item.id} className="knowledge-card">
                  <div className="knowledge-card-head">
                    <div>
                      <h4>{item.filename}</h4>
                      <p className="muted">{item.book_title} / {item.doc_type} / {item.allowed_role}</p>
                    </div>
                    <button className="danger-button" type="button" onClick={() => handleDelete(item.id)}>
                      删除
                    </button>
                  </div>
                  <div className="meta-row">
                    <span className="pill">{item.exists_on_disk ? '文件存在' : '文件缺失'}</span>
                    <span className="muted">导入人：{item.created_by}</span>
                    <span className="muted">{item.created_at}</span>
                  </div>
                  <p className="mono-text">{item.file_path}</p>
                </article>
              ))}
            </div>
          ) : (
            <div className="empty-box muted">还没有符合筛选条件的知识文档。</div>
          )}
        </div>
      </div>
    </section>
  )
}
