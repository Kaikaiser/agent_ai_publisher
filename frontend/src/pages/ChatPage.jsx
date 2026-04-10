import { useEffect, useState } from 'react'

import { apiRequest } from '../api/client'

function LoadingState() {
  return (
    <div className="status-box">
      <div className="double-ring-spinner" aria-hidden="true" />
      <div>
        <p className="status-title">正在载入开霸霸数字资产...</p>
        <p className="status-caption">系统正在连接知识库、检索上下文并准备生成回答，请稍候。</p>
      </div>
    </div>
  )
}

function EmptyState() {
  return (
    <div className="assistant-placeholder">
      <div className="assistant-placeholder-mark">AI</div>
      <h3>从一个问题开始</h3>
      <p>你可以直接输入图书内容相关问题，或上传题目图片后先识别、再确认问答。</p>
      <div className="assistant-placeholder-tags">
        <span className="pill">图书问答</span>
        <span className="pill">拍照搜题</span>
        <span className="pill">引用来源</span>
      </div>
    </div>
  )
}

export default function ChatPage() {
  const [question, setQuestion] = useState('')
  const [bookTitle, setBookTitle] = useState('')
  const [docType, setDocType] = useState('')
  const [imageFile, setImageFile] = useState(null)
  const [imagePreview, setImagePreview] = useState('')
  const [recognizedText, setRecognizedText] = useState('')
  const [recognizeSuccess, setRecognizeSuccess] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!imageFile) {
      setImagePreview('')
      return
    }

    const nextPreview = URL.createObjectURL(imageFile)
    setImagePreview(nextPreview)

    return () => {
      URL.revokeObjectURL(nextPreview)
    }
  }, [imageFile])

  const handleSubmit = async (event) => {
    event.preventDefault()
    setError('')
    setLoading(true)
    try {
      const data = await apiRequest('/chat/ask', {
        method: 'POST',
        body: JSON.stringify({
          question,
          book_title: bookTitle || null,
          doc_type: docType || null,
        }),
      })
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleRecognizeImage = async (event) => {
    event.preventDefault()
    setError('')
    setRecognizeSuccess(false)
    if (!imageFile) {
      setError('请先上传题目图片。')
      return
    }
    const formData = new FormData()
    formData.append('file', imageFile)
    setLoading(true)
    try {
      const data = await apiRequest('/chat/recognize-image', {
        method: 'POST',
        body: formData,
      })
      setRecognizedText(data.recognized_text)
      setRecognizeSuccess(true)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleRecognizedAsk = async (event) => {
    event.preventDefault()
    setError('')
    if (!recognizedText.trim()) {
      setError('请先识别图片内容，或手动补充题目文本。')
      return
    }
    setLoading(true)
    try {
      const data = await apiRequest('/chat/ask', {
        method: 'POST',
        body: JSON.stringify({
          question: recognizedText,
          book_title: bookTitle || null,
          doc_type: docType || null,
        }),
      })
      setResult({ ...data, recognized_text: recognizedText })
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="page-section">
      <div className="page-intro">
        <h2>书籍配套AI</h2>
        <p>将文本问答与拍照搜题整合到一个工作台中，输出答案时优先展示知识依据和引用来源。</p>
      </div>

      <div className="chat-workspace">
        <div className="chat-main stack">
          <div className="panel feature-panel feature-panel-primary">
            <div className="feature-panel-head">
              <div>
                <h3>文本问答</h3>
                <p className="muted">适合直接咨询教材、教辅、图书内容问题，也支持按书名和文档类型缩小检索范围。</p>
              </div>
              <span className="feature-badge">Text QA</span>
            </div>

            <form onSubmit={handleSubmit} className="stack">
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="例如：这本教辅适合几年级？请结合书内内容给出依据。"
                rows={6}
              />
              <div className="grid-two">
                <input value={bookTitle} onChange={(e) => setBookTitle(e.target.value)} placeholder="书名过滤，如：小学数学" />
                <input value={docType} onChange={(e) => setDocType(e.target.value)} placeholder="文档类型过滤，如：textbook" />
              </div>
              <div className="button-row">
                <button className="primary-button" type="submit">提交问答</button>
              </div>
            </form>
          </div>

          <div className="panel feature-panel feature-panel-soft">
            <div className="feature-panel-head">
              <div>
                <h3>拍照搜题</h3>
                <p className="muted">先上传题目图片并做 OCR 识别，再确认识别文本后继续走现有问答链路。</p>
              </div>
              <span className="feature-badge">OCR + QA</span>
            </div>

            <div className="photo-layout">
              <div className="stack">
                <div className="photo-flow-card">
                  <div className="photo-flow-step">
                    <span className="photo-step-index">1</span>
                    <div>
                      <strong>上传图片</strong>
                      <p className="muted">支持题目截图、教材照片或作业拍照。</p>
                    </div>
                  </div>
                  <form onSubmit={handleRecognizeImage} className="stack">
                    <input type="file" accept="image/*" onChange={(e) => setImageFile(e.target.files?.[0] || null)} />
                    <div className="button-row">
                      <button className="secondary-button" type="submit">识别题目</button>
                    </div>
                  </form>
                </div>

                <div className="photo-flow-card photo-flow-card-dashed">
                  <div className="photo-flow-step">
                    <span className="photo-step-index">2</span>
                    <div>
                      <strong>确认识别结果</strong>
                      <p className="muted">你可以先修改错别字或补足题干，再发起正式问答。</p>
                    </div>
                  </div>
                  {recognizeSuccess && (
                    <div className="recognize-banner">
                      <strong>识别完成</strong>
                      <span>已成功抽取题目文本，你可以先检查后再提交问答。</span>
                    </div>
                  )}
                  <form onSubmit={handleRecognizedAsk} className="stack">
                    <textarea
                      value={recognizedText}
                      onChange={(e) => setRecognizedText(e.target.value)}
                      className={recognizeSuccess ? 'textarea-highlight' : ''}
                      placeholder="识别结果会显示在这里。你可以先修正题干，再提交问答。"
                      rows={7}
                    />
                    <div className="button-row">
                      <button className="primary-button" type="submit">确认题目并问答</button>
                      <button className="ghost-button" type="button" onClick={() => { setRecognizedText(''); setRecognizeSuccess(false) }}>清空识别结果</button>
                    </div>
                  </form>
                </div>
              </div>

              <div className="photo-preview-card">
                <div className="photo-preview-head">
                  <strong>图片预览</strong>
                  <span className="muted">识别前先确认上传内容是否清晰</span>
                </div>
                {imagePreview ? (
                  <div className="photo-preview-frame">
                    <img src={imagePreview} alt="题目预览" className="photo-preview-image" />
                  </div>
                ) : (
                  <div className="photo-preview-empty">
                    <div className="photo-preview-empty-mark">图</div>
                    <p>上传题目图片后，这里会显示预览。</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="chat-side panel result-shell">
          {loading ? <LoadingState /> : result ? (
            <div className="result-card">
              <div className="result-head">
                <span className="pill">{result.grounded ? '已基于知识库作答' : '未找到充分依据'}</span>
              </div>
              {result.recognized_text && (
                <div className="result-block">
                  <h3>识别后的题目文本</h3>
                  <p>{result.recognized_text}</p>
                </div>
              )}
              <div className="result-block">
                <h3>AI 回答</h3>
                <p>{result.answer}</p>
              </div>
              <div className="result-block">
                <h4>引用来源</h4>
                <ul className="source-list">
                  {result.sources.map((source, index) => (
                    <li className="source-item" key={`${source.filename}-${index}`}>
                      <div className="source-item-head">
                        <strong>{source.filename}</strong>
                        {source.document_id && <span className="soft-tag">文档 #{source.document_id}</span>}
                      </div>
                      <p className="muted">{source.book_title} / {source.doc_type} / {source.location}</p>
                      <p>{source.preview}</p>
                      {source.content !== source.preview && (
                        <details className="source-details">
                          <summary>查看完整片段</summary>
                          <p>{source.content}</p>
                        </details>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ) : <EmptyState />}
        </div>
      </div>

      {error && (
        <div className="panel">
          <p className="error-text">{error}</p>
        </div>
      )}
    </section>
  )
}
