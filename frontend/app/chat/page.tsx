'use client'

import { useState, useRef, useEffect } from 'react'
import { Send, Loader2, FileText, File } from 'lucide-react'
import { chatApi, documentApi } from '@/lib/api'
import { formatCurrency } from '@/lib/utils'

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: any[]
  metrics?: any
  timestamp: Date
  isError?: boolean
  canRetry?: boolean
  originalQuery?: string
}

interface SourceDocument {
  content: string
  metadata: any
  score?: number
  document_name?: string
  page_number?: number
  chunk_index?: number
  confidence_score?: number
  citation_text?: string
}

interface Document {
  id: number
  file_name: string
  parsing_status: string
  fund_id?: number
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [conversationId, setConversationId] = useState<string>()
  const [documents, setDocuments] = useState<Document[]>([])
  const [selectedDocumentId, setSelectedDocumentId] = useState<number>()
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const handleRetry = async (originalQuery: string) => {
    if (loading) return

    setLoading(true)

    try {
      const response = await chatApi.query(originalQuery, undefined, conversationId, selectedDocumentId)

      // Replace the error message with success message
      setMessages(prev => prev.map(msg =>
        msg.originalQuery === originalQuery && msg.isError
          ? {
              role: 'assistant',
              content: response.answer,
              sources: response.sources,
              metrics: response.metrics,
              timestamp: new Date()
            }
          : msg
      ))
    } catch (error: any) {
      // Update the error message (retry failed again)
      setMessages(prev => prev.map(msg =>
        msg.originalQuery === originalQuery && msg.isError
          ? {
              ...msg,
              content: `Sorry, I encountered an error: ${error.response?.data?.detail || error.message}`,
              timestamp: new Date()
            }
          : msg
      ))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    // Load documents and create conversation on mount
    const loadData = async () => {
      try {
        const docs = await documentApi.list()
        setDocuments(docs)

        const conv = await chatApi.createConversation()
        setConversationId(conv.conversation_id)
      } catch (error) {
        console.error('Error loading data:', error)
      }
    }
    loadData()
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || loading) return

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    const currentQuery = input
    setInput('')
    setLoading(true)

    try {
      const response = await chatApi.query(currentQuery, undefined, conversationId, selectedDocumentId)

      const assistantMessage: Message = {
        role: 'assistant',
        content: response.answer,
        sources: response.sources,
        metrics: response.metrics,
        timestamp: new Date()
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (error: any) {
      const isNetworkError = error.code === 'NETWORK_ERROR' ||
                            error.message?.includes('Network Error') ||
                            error.message?.includes('Failed to fetch') ||
                            !error.response

      const errorMessage: Message = {
        role: 'assistant',
        content: `Sorry, I encountered an error: ${error.response?.data?.detail || error.message}`,
        timestamp: new Date(),
        isError: true,
        canRetry: isNetworkError,
        originalQuery: currentQuery
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-5xl mx-auto h-[calc(100vh-12rem)]">
      <div className="mb-4">
        <h1 className="text-4xl font-bold mb-2">Fund Analysis Chat</h1>
        <p className="text-gray-600">
          Ask questions about fund performance, metrics, and transactions
        </p>

        {/* Document Selector */}
        {documents.length > 0 && (
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select a document to chat about:
            </label>
            <select
              value={selectedDocumentId || ''}
              onChange={(e) => setSelectedDocumentId(e.target.value ? parseInt(e.target.value) : undefined)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">General questions (no specific document)</option>
              {documents.map((doc) => (
                <option key={doc.id} value={doc.id}>
                  {doc.file_name} {doc.parsing_status === 'completed' ? '✅' : doc.parsing_status === 'processing' ? '⏳' : '❌'}
                </option>
              ))}
            </select>
            {selectedDocumentId && (
              <p className="text-xs text-gray-500 mt-1">
                Chatting about: {documents.find(d => d.id === selectedDocumentId)?.file_name}
              </p>
            )}
          </div>
        )}
      </div>

      <div className="bg-white rounded-lg shadow-md flex flex-col h-full">
        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.length === 0 && (
            <div className="text-center py-12">
              <div className="text-gray-400 mb-4">
                <File className="w-16 h-16 mx-auto" />
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                Start a conversation
              </h3>
              <p className="text-gray-600 mb-6">
                {selectedDocumentId
                  ? `Ask questions about ${documents.find(d => d.id === selectedDocumentId)?.file_name}:`
                  : "Select a document above or try general questions:"
                }
              </p>
              <div className="space-y-2 max-w-md mx-auto">
                {selectedDocumentId ? (
                  <>
                    <SampleQuestion
                      question="What is the current DPI?"
                      onClick={() => setInput("What is the current DPI?")}
                    />
                    <SampleQuestion
                      question="Calculate the IRR for this fund"
                      onClick={() => setInput("Calculate the IRR for this fund")}
                    />
                    <SampleQuestion
                      question="Show me all capital calls"
                      onClick={() => setInput("Show me all capital calls")}
                    />
                    <SampleQuestion
                      question="What does Paid-In Capital mean?"
                      onClick={() => setInput("What does Paid-In Capital mean?")}
                    />
                  </>
                ) : (
                  <>
                    <SampleQuestion
                      question="What is DPI?"
                      onClick={() => setInput("What is DPI?")}
                    />
                    <SampleQuestion
                      question="How is IRR calculated?"
                      onClick={() => setInput("How is IRR calculated?")}
                    />
                    <SampleQuestion
                      question="What does Paid-In Capital mean?"
                      onClick={() => setInput("What does Paid-In Capital mean?")}
                    />
                  </>
                )}
              </div>
            </div>
          )}

          {messages.map((message, index) => (
            <MessageBubble
              key={index}
              message={message}
              onRetry={handleRetry}
              isLoading={loading}
            />
          ))}

          {loading && (
            <div className="flex items-center space-x-2 text-gray-500">
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Thinking...</span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="border-t p-4">
          <form onSubmit={handleSubmit} className="flex space-x-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about the fund..."
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
            >
              <Send className="w-4 h-4" />
              <span>Send</span>
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}

function MessageBubble({ message, onRetry, isLoading }: {
  message: Message;
  onRetry?: (query: string) => void;
  isLoading?: boolean;
}) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-3xl ${isUser ? 'ml-12' : 'mr-12'}`}>
        <div
          className={`rounded-lg p-4 ${
            isUser
              ? 'bg-blue-600 text-white'
              : message.isError
              ? 'bg-red-100 text-red-900 border border-red-300'
              : 'bg-gray-100 text-gray-900'
          }`}
        >
          <p className="whitespace-pre-wrap">{message.content}</p>
          {message.canRetry && message.originalQuery && onRetry && (
            <div className="mt-3 pt-3 border-t border-red-200">
              <button
                onClick={() => onRetry(message.originalQuery!)}
                disabled={isLoading}
                className="px-3 py-1 bg-red-600 text-white text-sm rounded hover:bg-red-700 transition disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? 'Retrying...' : 'Retry'}
              </button>
            </div>
          )}
        </div>

        {/* Metrics Display */}
        {message.metrics && (
          <div className="mt-3 bg-white border border-gray-200 rounded-lg p-4">
            <h4 className="font-semibold text-sm text-gray-700 mb-2">Calculated Metrics</h4>
            <div className="grid grid-cols-2 gap-3">
              {Object.entries(message.metrics).map(([key, value]) => {
                if (value === null || value === undefined) return null
                
                let displayValue: string
                if (typeof value === 'number' && key.includes('irr')) {
                  displayValue = `${value.toFixed(2)}%`
                } else if (typeof value === 'number') {
                  displayValue = formatCurrency(value)
                } else {
                  displayValue = String(value)
                }
                
                return (
                  <div key={key} className="text-sm">
                    <span className="text-gray-600">{key.toUpperCase()}:</span>{' '}
                    <span className="font-semibold">{displayValue}</span>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* Sources Display */}
        {message.sources && message.sources.length > 0 && (
          <div className="mt-3">
            <details className="bg-white border border-gray-200 rounded-lg">
              <summary className="px-4 py-2 cursor-pointer text-sm font-medium text-gray-700 hover:bg-gray-50">
                View Sources ({message.sources.length})
              </summary>
              <div className="px-4 py-3 space-y-3 border-t">
                {message.sources.slice(0, 3).map((source: SourceDocument, idx) => (
                  <div key={idx} className="text-xs bg-gray-50 p-3 rounded border">
                    {/* Enhanced Citation Header */}
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <FileText className="w-3 h-3 text-blue-600" />
                        <span className="font-medium text-blue-900">
                          {source.citation_text || source.document_name || `Source ${idx + 1}`}
                        </span>
                      </div>
                      {source.confidence_score && (
                        <span className="text-green-600 font-medium">
                          {(source.confidence_score * 100).toFixed(0)}% confidence
                        </span>
                      )}
                    </div>

                    {/* Source Content */}
                    <p className="text-gray-700 line-clamp-3 mb-2">{source.content}</p>

                    {/* Metadata */}
                    <div className="flex items-center justify-between text-gray-500">
                      <div className="flex items-center space-x-3">
                        {source.page_number && (
                          <span>Page {source.page_number}</span>
                        )}
                        {source.chunk_index !== undefined && (
                          <span>Section {source.chunk_index + 1}</span>
                        )}
                      </div>
                      {source.score && (
                        <span>Relevance: {(source.score * 100).toFixed(0)}%</span>
                      )}
                    </div>
                  </div>
                ))}

                {/* Show more indicator */}
                {message.sources.length > 3 && (
                  <div className="text-center text-xs text-gray-500 py-2">
                    And {message.sources.length - 3} more sources...
                  </div>
                )}
              </div>
            </details>
          </div>
        )}

        <p className="text-xs text-gray-500 mt-2">
          {message.timestamp.toLocaleTimeString()}
        </p>
      </div>
    </div>
  )
}

function SampleQuestion({ question, onClick }: { question: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="w-full text-left px-4 py-2 bg-gray-50 hover:bg-gray-100 rounded-lg text-sm text-gray-700 transition"
    >
      &quot;{question}&quot;
    </button>
  )
}
