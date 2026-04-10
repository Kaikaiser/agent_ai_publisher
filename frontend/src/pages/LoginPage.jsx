import { Navigate, useNavigate } from 'react-router-dom'
import { useState } from 'react'

import { apiRequest } from '../api/client'
import { saveSession, getCurrentUser } from '../auth/session'

export default function LoginPage() {
  const navigate = useNavigate()
  const [username, setUsername] = useState('admin')
  const [password, setPassword] = useState('admin123456')
  const [error, setError] = useState('')

  if (getCurrentUser()) {
    return <Navigate to="/chat" replace />
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    setError('')
    try {
      const data = await apiRequest('/auth/login', {
        method: 'POST',
        body: JSON.stringify({ username, password }),
      })
      saveSession(data)
      navigate('/chat')
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <div className="auth-page">
      <form className="panel auth-panel" onSubmit={handleSubmit}>
        <div className="auth-brand">
          <div className="auth-brand-mark">开</div>
          <div>
            <h2>开霸霸</h2>
            <p>AI 读书助手后台登录</p>
          </div>
        </div>
        <div className="stack">
          <input value={username} onChange={(e) => setUsername(e.target.value)} placeholder="请输入账号" />
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="请输入密码" />
          {error && <p className="error-text">{error}</p>}
          <button className="primary-button" type="submit">登录系统</button>
        </div>
      </form>
    </div>
  )
}