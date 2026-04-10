import { Navigate, NavLink, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { useMemo, useState } from 'react'

import { getCurrentUser, logout } from './auth/session'
import LoginPage from './pages/LoginPage'
import ChatPage from './pages/ChatPage'
import KnowledgePage from './pages/KnowledgePage'
import HistoryPage from './pages/HistoryPage'

const MENU_ITEMS = [
  { label: '书籍配套AI', path: '/chat', icon: '书' },
  { label: '一书一码', path: '/knowledge', icon: '码', adminOnly: true },
  { label: '开霸霸书库', path: '/history', icon: '库' },
]

function ProtectedLayout() {
  const navigate = useNavigate()
  const location = useLocation()
  const [user, setUser] = useState(getCurrentUser())

  if (!user) {
    return <Navigate to="/login" replace />
  }

  const activeMenu = useMemo(() => {
    return MENU_ITEMS.find((item) => location.pathname.startsWith(item.path))?.label || '书籍配套AI'
  }, [location.pathname])

  const handleLogout = () => {
    logout()
    setUser(null)
    navigate('/login')
  }

  return (
    <div className="admin-shell">
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div className="sidebar-logo-mark">开</div>
          <span>开霸霸</span>
        </div>

        <nav className="sidebar-nav">
          {MENU_ITEMS.filter((item) => !item.adminOnly || user.role === 'admin').map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) => `menu-item${isActive ? ' active' : ''}`}
            >
              <span className="menu-item-icon" aria-hidden="true">{item.icon}</span>
              <span>{item.label}</span>
              <span className="menu-item-rail" aria-hidden="true" />
            </NavLink>
          ))}
        </nav>
      </aside>

      <div className="workspace">
        <header className="top-header">
          <div className="top-header-meta">
            <div>
              <h1 className="top-header-title">{activeMenu}</h1>
              <p className="top-header-subtitle">开霸霸 AI 读书助手，为出版社内容问答、图书检索与数字资产运营提供统一工作台。</p>
            </div>
          </div>

          <div className="top-header-actions">
            <div className="header-search">
              <span className="header-search-icon" aria-hidden="true">搜</span>
              <input placeholder="搜索书名、知识点或历史问答" />
            </div>
            <button className="header-avatar" onClick={handleLogout} title="点击退出登录">
              管理员
            </button>
          </div>
        </header>

        <main className="workspace-main">
          <Routes>
            <Route path="/chat" element={<ChatPage />} />
            <Route path="/knowledge" element={user.role === 'admin' ? <KnowledgePage /> : <Navigate to="/chat" replace />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="*" element={<Navigate to="/chat" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/*" element={<ProtectedLayout />} />
    </Routes>
  )
}