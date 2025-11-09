'use client'

import { useQuery } from '@tanstack/react-query'
import { useParams } from 'next/navigation'
import { fundApi } from '@/lib/api'
import { formatCurrency, formatPercentage, formatDate } from '@/lib/utils'
import { Loader2, TrendingUp, DollarSign, Calendar } from 'lucide-react'

export default function FundDetailPage() {
  const params = useParams()
  const fundId = parseInt(params.id as string)

  const { data: fund, isLoading: fundLoading } = useQuery({
    queryKey: ['fund', fundId],
    queryFn: () => fundApi.get(fundId)
  })

  const { data: capitalCalls } = useQuery({
    queryKey: ['transactions', fundId, 'capital_calls'],
    queryFn: () => fundApi.getTransactions(fundId, 'capital_calls', 1, 10)
  })

  const { data: distributions } = useQuery({
    queryKey: ['transactions', fundId, 'distributions'],
    queryFn: () => fundApi.getTransactions(fundId, 'distributions', 1, 10)
  })

  if (fundLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
      </div>
    )
  }

  if (!fund) {
    return <div>Fund not found</div>
  }

  const metrics = fund.metrics || {}

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">{fund.name}</h1>
        <div className="flex items-center space-x-4 text-gray-600">
          {fund.gp_name && <span>GP: {fund.gp_name}</span>}
          {fund.vintage_year && <span>Vintage: {fund.vintage_year}</span>}
          {fund.fund_type && <span>Type: {fund.fund_type}</span>}
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <MetricCard
          title="DPI"
          value={metrics.dpi?.toFixed(2) + 'x' || 'N/A'}
          description="Distribution to Paid-In"
          icon={<TrendingUp className="w-6 h-6" />}
          color="blue"
        />
        <MetricCard
          title="IRR"
          value={metrics.irr ? formatPercentage(metrics.irr) : 'N/A'}
          description="Internal Rate of Return"
          icon={<TrendingUp className="w-6 h-6" />}
          color="green"
        />
        <MetricCard
          title="Paid-In Capital"
          value={metrics.pic ? formatCurrency(metrics.pic) : 'N/A'}
          description="Total capital called"
          icon={<DollarSign className="w-6 h-6" />}
          color="purple"
        />
        <MetricCard
          title="Distributions"
          value={metrics.total_distributions ? formatCurrency(metrics.total_distributions) : 'N/A'}
          description="Total distributions"
          icon={<DollarSign className="w-6 h-6" />}
          color="orange"
        />
      </div>

      {/* Transactions Tables */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Capital Calls */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4">Recent Capital Calls</h2>
          {capitalCalls && capitalCalls.items.length > 0 ? (
            <div className="space-y-3">
              {capitalCalls.items.map((call: any) => (
                <TransactionRow
                  key={call.id}
                  date={call.call_date}
                  type={call.call_type}
                  amount={call.amount}
                  isNegative
                />
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-sm">No capital calls found</p>
          )}
        </div>

        {/* Distributions */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4">Recent Distributions</h2>
          {distributions && distributions.items.length > 0 ? (
            <div className="space-y-3">
              {distributions.items.map((dist: any) => (
                <TransactionRow
                  key={dist.id}
                  date={dist.distribution_date}
                  type={dist.distribution_type}
                  amount={dist.amount}
                  isRecallable={dist.is_recallable}
                />
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-sm">No distributions found</p>
          )}
        </div>
      </div>
    </div>
  )
}

function MetricCard({ title, value, description, icon, color }: {
  title: string
  value: string
  description: string
  icon: React.ReactNode
  color: 'blue' | 'green' | 'purple' | 'orange'
}) {
  const colorClasses = {
    blue: 'bg-blue-100 text-blue-600',
    green: 'bg-green-100 text-green-600',
    purple: 'bg-purple-100 text-purple-600',
    orange: 'bg-orange-100 text-orange-600',
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className={`w-12 h-12 rounded-lg ${colorClasses[color]} flex items-center justify-center mb-4`}>
        {icon}
      </div>
      <h3 className="text-sm font-medium text-gray-600 mb-1">{title}</h3>
      <p className="text-2xl font-bold text-gray-900 mb-1">{value}</p>
      <p className="text-xs text-gray-500">{description}</p>
    </div>
  )
}

function TransactionRow({ date, type, amount, isNegative, isRecallable }: {
  date: string
  type: string
  amount: number
  isNegative?: boolean
  isRecallable?: boolean
}) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-gray-100 last:border-0">
      <div className="flex-1">
        <p className="text-sm font-medium text-gray-900">{type}</p>
        <div className="flex items-center space-x-2 mt-1">
          <Calendar className="w-3 h-3 text-gray-400" />
          <p className="text-xs text-gray-500">{formatDate(date)}</p>
          {isRecallable && (
            <span className="text-xs bg-yellow-100 text-yellow-800 px-2 py-0.5 rounded">
              Recallable
            </span>
          )}
        </div>
      </div>
      <p className={`text-sm font-semibold ${isNegative ? 'text-red-600' : 'text-green-600'}`}>
        {isNegative ? '-' : '+'}{formatCurrency(Math.abs(amount))}
      </p>
    </div>
  )
}
