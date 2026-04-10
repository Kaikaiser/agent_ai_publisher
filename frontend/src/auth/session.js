export function saveSession(payload) {
  localStorage.setItem('token', payload.access_token)
  localStorage.setItem('user', JSON.stringify(payload.user))
}

export function getCurrentUser() {
  const raw = localStorage.getItem('user')
  return raw ? JSON.parse(raw) : null
}

export function logout() {
  localStorage.removeItem('token')
  localStorage.removeItem('user')
}
