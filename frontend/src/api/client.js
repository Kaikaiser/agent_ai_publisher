export const API_BASE_URL = 'http://127.0.0.1:8000/api'

const STATUS_MESSAGES = {
  400: '请求参数不正确，请检查后重试。',
  401: '登录已失效或账号密码错误，请重新登录。',
  403: '当前账号没有权限执行这个操作。',
  404: '请求的接口或资源不存在。',
  409: '数据状态冲突，请刷新页面后重试。',
  422: '提交的数据格式不正确，请检查输入内容。',
  500: '服务器处理请求时发生错误，请稍后再试。',
  502: '后端服务暂时不可用，请稍后再试。',
  503: '服务正在维护或启动中，请稍后再试。',
}

function toChineseDetail(detail, status) {
  if (!detail) {
    return STATUS_MESSAGES[status] || '请求失败，请稍后再试。'
  }

  if (typeof detail === 'string') {
    const normalized = detail.trim()

    if (!normalized) {
      return STATUS_MESSAGES[status] || '请求失败，请稍后再试。'
    }

    const lower = normalized.toLowerCase()
    if (lower.includes('invalid credentials') || lower.includes('invalid username or password')) {
      return '账号或密码错误，请重新输入。'
    }
    if (lower.includes('missing token')) {
      return '当前未登录，请先登录后再试。'
    }
    if (lower.includes('not authenticated') || lower.includes('could not validate credentials') || lower.includes('invalid token')) {
      return '当前未登录或登录已失效，请重新登录。'
    }
    if (lower.includes('user not found')) {
      return '当前用户不存在，请重新登录。'
    }
    if (lower.includes('not enough permissions') || lower.includes('admin role required')) {
      return '当前账号没有权限执行这个操作。'
    }
    if (lower.includes('openai api key')) {
      return '服务端尚未配置 OpenAI API Key，请先检查后端环境变量。'
    }
    if (lower.includes('unsupported file type')) {
      return '暂不支持该文件类型，请上传 TXT、Markdown、PDF 或 DOCX 文件。'
    }
    if (lower.includes('only image uploads are supported')) {
      return '仅支持上传图片文件。'
    }
    if (lower.includes('uploaded image is empty')) {
      return '上传的图片为空，请重新选择。'
    }
    if (lower.includes('image file')) {
      return '请上传图片文件后再试。'
    }

    return normalized
  }

  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (typeof item === 'string') {
          return item
        }
        const path = Array.isArray(item?.loc) ? item.loc.join(' / ') : '字段'
        return `${path}：${item?.msg || '输入不合法'}`
      })
      .join('；')
  }

  return STATUS_MESSAGES[status] || '请求失败，请稍后再试。'
}

function normalizeNetworkError(error) {
  const message = String(error?.message || '').toLowerCase()
  if (message.includes('failed to fetch')) {
    return '无法连接后端服务，请确认后端已经启动，且地址与端口配置正确。'
  }
  if (message.includes('networkerror')) {
    return '网络请求失败，请检查当前网络或后端服务状态。'
  }
  if (message.includes('load failed')) {
    return '请求发送失败，请稍后再试。'
  }
  return '请求发送失败，请检查网络或稍后重试。'
}

export async function apiRequest(path, options = {}) {
  const token = localStorage.getItem('token')
  const headers = {
    ...(options.headers || {}),
  }

  if (!(options.body instanceof FormData)) {
    headers['Content-Type'] = headers['Content-Type'] || 'application/json'
  }

  if (token) {
    headers.Authorization = `Bearer ${token}`
  }

  let response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, { ...options, headers })
  } catch (error) {
    throw new Error(normalizeNetworkError(error))
  }

  if (!response.ok) {
    const data = await response.json().catch(() => ({ detail: null }))
    const message = toChineseDetail(data?.detail, response.status)

    if (response.status === 401) {
      localStorage.removeItem('token')
      localStorage.removeItem('user')
    }

    throw new Error(message)
  }

  return response.json()
}