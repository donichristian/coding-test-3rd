'use client'

import { useQuery } from '@tanstack/react-query'
import Link from 'next/link'
import { fundApi } from '@/lib/api'
import { formatCurrency, formatPercentage } from '@/lib/utils'
import { TrendingUp, TrendingDown, ArrowRight, Loader2 } from 'lucide-react'

export default function FundsPage() {
  const { data: funds, isLoading, error } = useQuery({
    queryKey: ['funds'],
    queryFn: () => fundApi.list()
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-800">Error loading funds: {(error as Error).message}</p>
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold mb-2">Fund Portfolio</h1>
          <p className="text-gray-600">
            View and analyze your fund investments
          </p>
        </div>
        <Link
          href="/upload"
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
        >
          Add New Fund
        </Link>
      </div>

      {funds && funds.length === 0 ? (
        <div className="bg-white rounded-lg shadow-md p-12 text-center">
          <p className="text-gray-600 mb-4">No funds found. Upload a fund document to get started.</p>
          <Link
            href="/upload"
            className="inline-block px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
          >
            Upload Document
          </Link>
        </div>
      ) : (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {funds?.map((fund: any) => (
            <FundCard key={fund.id} fund={fund} />
          ))}
        </div>
      )}
    </div>
  )
}

function FundCard({ fund }: { fund: any }) {
  const metrics = fund.metrics || {}
  const dpi = metrics.dpi || 0
  const irr = metrics.irr || 0

  return (
    <Link href={`/funds/${fund.id}`}>
      <div className="bg-white rounded-lg shadow-md hover:shadow-lg transition p-6 h-full">
        <div className="mb-4">
          <h3 className="text-xl font-semibold text-gray-900 mb-1">
            {fund.name}
          </h3>
          {fund.gp_name && (
            <p className="text-sm text-gray-600">GP: {fund.gp_name}</p>
          )}
          {fund.vintage_year && (
            <p className="text-sm text-gray-500">Vintage: {fund.vintage_year}</p>
          )}
        </div>

        <div className="space-y-3 mb-4">
          <MetricRow
            label="DPI"
            value={dpi.toFixed(2) + 'x'}
            positive={dpi >= 1}
          />
          <MetricRow
            label="IRR"
            value={formatPercentage(irr)}
            positive={irr >= 0}
          />
          {metrics.pic > 0 && (
            <MetricRow
              label="Paid-In Capital"
              value={formatCurrency(metrics.pic)}
            />
          )}
        </div>

        <div className="flex items-center text-blue-600 text-sm font-medium">
          View Details <ArrowRight className="w-4 h-4 ml-1" />
        </div>
      </div>
    </Link>
  )
}

function MetricRow({ label, value, positive }: { 
  label: string
  value: string
  positive?: boolean 
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-sm text-gray-600">{label}</span>
      <div className="flex items-center space-x-1">
        <span className="font-semibold text-gray-900">{value}</span>
        {positive !== undefined && (
          positive ? (
            <TrendingUp className="w-4 h-4 text-green-600" />
          ) : (
            <TrendingDown className="w-4 h-4 text-red-600" />
          )
        )}
      </div>
    </div>
  )
}
