import { useEffect, useState } from 'react'

import { apiRequest } from '../api/client'

function buildHistoryQuery(filters) {
  const params = new URLSearchParams()
  if (filters.query.trim()) params.set('query', filters.query.trim())
  if (filters.bookTitle.trim()) params.set('book_title', filters.bookTitle.trim())
  if (filters.docType.trim()) params.set('doc_type', filters.docType.trim())
  if (filters.grounded !== 'all') params.set('grounded', filters.grounded)
  const query = params.toString()
  return query ? `/conversations?${query}` : '/conversations'
}

export default function HistoryPage() {
  const [items, setItems] = useState([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [filters, setFilters] = useState({ query: '', grounded: 'all', bookTitle: '', docType: '' })

  const loadItems = async (nextFilters = filters) => {
    setLoading(true)
    setError('')
    try {
      const data = await apiRequest(buildHistoryQuery(nextFilters))
      setItems(data.items)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadItems()
  }, [])

  const handleSubmit = async (event) => {
    event.preventDefault()
    await loadItems(filters)
  }

  return (
    <section className="page-section">
      <div className="page-intro">
        <h2>历史记录</h2>
        <p>按问题、依据状态、书名和文档类型回看历史问答，快速定位已生成的知识答案。</p>
      </div>

      <div className="panel">
        <form onSubmit={handleSubmit} className="stack">
          <div className="grid-four">
            <input
              value={filters.query}
              onChange={(e) => setFilters((current) => ({ ...current, query: e.target.value }))}
              placeholder="搜索问题、回答或来源摘要"
            />
            <select
              value={filters.grounded}
              onChange={(e) => setFilters((current) => ({ ...current, grounded: e.target.value }))}
            >
              <option value="all">全部回答</option>
              <option value="true">仅知识库作答</option>
              <option value="false">仅未命中依据</option>
            </select>
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
            <button className="primary-button" type="submit">查询历史</button>
            <button
              className="ghost-button"
              type="button"
              onClick={() => {
                const nextFilters = { query: '', grounded: 'all', bookTitle: '', docType: '' }
                setFilters(nextFilters)
                loadItems(nextFilters)
              }}
            >
              清空筛选
            </button>
          </div>
        </form>

        {error && <p className="error-text">{error}</p>}
        {loading ? (
          <div className="empty-box muted">正在加载历史记录...</div>
        ) : (
          <div className="history-list">
            {items.map((item) => (
              <article key={item.id} className="history-card">
                <div className="meta-row">
                  <p className="muted">{item.created_at}</p>
                  <span className="pill">{item.grounded ? '已基于知识库作答' : '未找到充分依据'}</span>
                </div>
                <h3>{item.question}</h3>
                <p>{item.answer}</p>
                <div className="tag-row">
                  {item.book_titles.map((bookTitle) => <span key={bookTitle} className="soft-tag">{bookTitle}</span>)}
                  {item.doc_types.map((docType) => <span key={docType} className="soft-tag">{docType}</span>)}
                  <span className="soft-tag">来源 {item.source_count} 条</span>
                </div>
                {item.source_preview && <p className="muted">来源摘要：{item.source_preview}</p>}
              </article>
            ))}
            {!items.length && <div className="empty-box muted">没有符合筛选条件的历史记录。</div>}
          </div>
        )}
      </div>
    </section>
  )
}
