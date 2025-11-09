import Link from 'next/link'
import { ArrowRight, Upload, MessageSquare, BarChart3, FileText } from 'lucide-react'

export default function Home() {
  return (
    <div className="max-w-6xl mx-auto">
      {/* Hero Section */}
      <div className="text-center mb-16">
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          Fund Performance Analysis System
        </h1>
        <p className="text-xl text-gray-600 mb-8">
          AI-powered fund analysis with intelligent document processing and natural language queries
        </p>
        <div className="flex gap-4 justify-center">
          <Link
            href="/upload"
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition flex items-center gap-2"
          >
            <Upload className="w-5 h-5" />
            Upload Document
          </Link>
          <Link
            href="/chat"
            className="px-6 py-3 bg-white border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition flex items-center gap-2"
          >
            <MessageSquare className="w-5 h-5" />
            Start Chat
          </Link>
        </div>
      </div>

      {/* Features Grid */}
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-16">
        <FeatureCard
          icon={<Upload className="w-8 h-8 text-blue-600" />}
          title="Document Upload"
          description="Upload PDF fund reports and automatically extract structured data"
          href="/upload"
        />
        <FeatureCard
          icon={<MessageSquare className="w-8 h-8 text-green-600" />}
          title="AI Chat"
          description="Ask questions about your funds using natural language"
          href="/chat"
        />
        <FeatureCard
          icon={<BarChart3 className="w-8 h-8 text-purple-600" />}
          title="Fund Dashboard"
          description="View fund metrics, charts, and performance analytics"
          href="/funds"
        />
        <FeatureCard
          icon={<FileText className="w-8 h-8 text-orange-600" />}
          title="Documents"
          description="Manage and track all uploaded fund documents"
          href="/documents"
        />
      </div>

      {/* How It Works */}
      <div className="bg-white rounded-lg shadow-md p-8 mb-16">
        <h2 className="text-3xl font-bold mb-8 text-center">How It Works</h2>
        <div className="grid md:grid-cols-3 gap-8">
          <Step
            number="1"
            title="Upload Documents"
            description="Upload your fund performance PDF reports. The system automatically parses tables and text."
          />
          <Step
            number="2"
            title="Automatic Processing"
            description="Tables are stored in SQL database. Text is embedded and stored in vector database for semantic search."
          />
          <Step
            number="3"
            title="Ask Questions"
            description="Use natural language to query fund metrics (DPI, IRR) or ask about specific transactions and definitions."
          />
        </div>
      </div>

      {/* Sample Questions */}
      <div className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg p-8">
        <h2 className="text-2xl font-bold mb-6">Sample Questions You Can Ask</h2>
        <div className="grid md:grid-cols-2 gap-4">
          <SampleQuestion question="What is the current DPI of this fund?" />
          <SampleQuestion question="Calculate the IRR for this fund" />
          <SampleQuestion question="What does Paid-In Capital mean?" />
          <SampleQuestion question="Show me all capital calls in 2024" />
          <SampleQuestion question="Has the fund returned all capital to LPs?" />
          <SampleQuestion question="What was the largest distribution?" />
        </div>
      </div>
    </div>
  )
}

function FeatureCard({ icon, title, description, href }: {
  icon: React.ReactNode
  title: string
  description: string
  href: string
}) {
  return (
    <Link href={href} className="block">
      <div className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition h-full">
        <div className="mb-4">{icon}</div>
        <h3 className="text-xl font-semibold mb-2">{title}</h3>
        <p className="text-gray-600 text-sm mb-4">{description}</p>
        <div className="flex items-center text-blue-600 text-sm font-medium">
          Learn more <ArrowRight className="w-4 h-4 ml-1" />
        </div>
      </div>
    </Link>
  )
}

function Step({ number, title, description }: {
  number: string
  title: string
  description: string
}) {
  return (
    <div className="text-center">
      <div className="w-12 h-12 bg-blue-600 text-white rounded-full flex items-center justify-center text-xl font-bold mx-auto mb-4">
        {number}
      </div>
      <h3 className="text-lg font-semibold mb-2">{title}</h3>
      <p className="text-gray-600 text-sm">{description}</p>
    </div>
  )
}

function SampleQuestion({ question }: { question: string }) {
  return (
    <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-200">
      <p className="text-gray-700 text-sm">&quot;{question}&quot;</p>
    </div>
  )
}
