import axios from 'axios'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// Always use localhost for browser requests - Docker networking is handled by port mapping
const FINAL_API_URL = API_URL

export const api = axios.create({
  baseURL: FINAL_API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: false, // Keep disabled for CORS simplicity
})

// Add request interceptor to handle CORS preflight
api.interceptors.request.use((config) => {
  // For file uploads, don't set Content-Type - let browser set it with boundary
  if (config.data instanceof FormData) {
    delete config.headers['Content-Type']
  }
  return config
})

// Remove CORS headers from client-side - let the server handle CORS

// Document APIs
export const documentApi = {
  upload: async (file: File, fundId?: number) => {
    const formData = new FormData()
    formData.append('file', file)
    if (fundId) {
      formData.append('fund_id', fundId.toString())
    }

    // Use fetch directly for file uploads to avoid axios CORS issues
    const response = await fetch(`${FINAL_API_URL}/api/documents/upload`, {
      method: 'POST',
      body: formData,
      // Don't set Content-Type header - let browser set it with boundary
    })

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.status} ${response.statusText}`)
    }

    return response.json()
  },
  
  getStatus: async (documentId: number) => {
    const response = await api.get(`/api/documents/${documentId}/status`)
    return response.data
  },

  getDetails: async (documentId: number) => {
    const response = await api.get(`/api/documents/${documentId}`)
    return response.data
  },
  
  list: async (fundId?: number) => {
    const params = fundId ? { fund_id: fundId } : {}
    const response = await api.get('/api/documents/', { params })
    return response.data
  },
  
  delete: async (documentId: number) => {
    const response = await api.delete(`/api/documents/${documentId}`)
    return response.data
  },
}

// Fund APIs
export const fundApi = {
  list: async () => {
    const response = await api.get('/api/funds/')
    return response.data
  },
  
  get: async (fundId: number) => {
    const response = await api.get(`/api/funds/${fundId}`)
    return response.data
  },
  
  create: async (fund: any) => {
    const response = await api.post('/api/funds/', fund)
    return response.data
  },
  
  getTransactions: async (fundId: number, type: string, page: number = 1, limit: number = 50) => {
    const response = await api.get(`/api/funds/${fundId}/transactions`, {
      params: { transaction_type: type, page, limit }
    })
    return response.data
  },
  
  getMetrics: async (fundId: number) => {
    const response = await api.get(`/api/funds/${fundId}/metrics`)
    return response.data
  },
}

// Chat APIs with retry functionality
export const chatApi = {
  query: async (query: string, fundId?: number, conversationId?: string, documentId?: number, retryCount: number = 0): Promise<any> => {
    const maxRetries = 3
    const baseDelay = 1000 // 1 second

    try {
      const response = await api.post('/api/chat/query', {
        query,
        fund_id: fundId,
        document_id: documentId,
        conversation_id: conversationId,
      })
      return response.data
    } catch (error: any) {
      // Check if it's a network error and we haven't exceeded max retries
      if (retryCount < maxRetries && (
        error.code === 'NETWORK_ERROR' ||
        error.message?.includes('Network Error') ||
        error.message?.includes('Failed to fetch') ||
        !error.response // No response means network issue
      )) {
        const delay = baseDelay * Math.pow(2, retryCount) // Exponential backoff
        console.log(`Network error, retrying in ${delay}ms... (attempt ${retryCount + 1}/${maxRetries})`)

        await new Promise(resolve => setTimeout(resolve, delay))

        return chatApi.query(query, fundId, conversationId, documentId, retryCount + 1)
      }

      // If max retries reached or it's not a network error, throw the error
      throw error
    }
  },
  
  createConversation: async (fundId?: number) => {
    const response = await api.post('/api/chat/conversations', {
      fund_id: fundId,
    })
    return response.data
  },
  
  getConversation: async (conversationId: string) => {
    const response = await api.get(`/api/chat/conversations/${conversationId}`)
    return response.data
  },
}

// Metrics APIs
export const metricsApi = {
  getFundMetrics: async (fundId: number, metric?: string) => {
    const params = metric ? { metric } : {}
    const response = await api.get(`/api/metrics/funds/${fundId}/metrics`, { params })
    return response.data
  },
}
