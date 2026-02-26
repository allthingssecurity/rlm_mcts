import { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import type { RubricNode } from '../types'

interface Props {
  node: RubricNode | null
  convergence: number[]
}

const GAUGES = [
  { key: 'reward_generalization', label: 'Generalization', weight: 1.0, color: '#22c55e' },
  { key: 'reward_calibration', label: 'Calibration', weight: 0.4, color: '#eab308' },
  { key: 'reward_discrimination', label: 'Discrimination', weight: 0.3, color: '#3b82f6' },
  { key: 'reward_validity', label: 'Validity', weight: 0.2, color: '#a855f7' },
  { key: 'reward_iteration', label: 'Iteration', weight: 0.2, color: '#06b6d4' },
]

export default function RewardDashboard({ node, convergence }: Props) {
  const chartRef = useRef<SVGSVGElement>(null)

  // Convergence line chart
  useEffect(() => {
    if (!chartRef.current || convergence.length < 2) return

    const svg = d3.select(chartRef.current)
    svg.selectAll('*').remove()

    const width = chartRef.current.clientWidth
    const height = chartRef.current.clientHeight
    const margin = { top: 10, right: 15, bottom: 25, left: 35 }
    const innerW = width - margin.left - margin.right
    const innerH = height - margin.top - margin.bottom

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    const x = d3.scaleLinear().domain([0, convergence.length - 1]).range([0, innerW])
    const y = d3.scaleLinear().domain([0, 1]).range([innerH, 0])

    // Grid
    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).ticks(Math.min(convergence.length, 10)).tickFormat(d => `${(d as number) + 1}`))
      .call(g => g.selectAll('text').attr('fill', '#6b7280').attr('font-size', '9px'))
      .call(g => g.selectAll('line').attr('stroke', '#374151'))
      .call(g => g.select('.domain').attr('stroke', '#374151'))

    g.append('g')
      .call(d3.axisLeft(y).ticks(5))
      .call(g => g.selectAll('text').attr('fill', '#6b7280').attr('font-size', '9px'))
      .call(g => g.selectAll('line').attr('stroke', '#374151'))
      .call(g => g.select('.domain').attr('stroke', '#374151'))

    // Line
    const line = d3.line<number>()
      .x((_, i) => x(i))
      .y(d => y(d))
      .curve(d3.curveMonotoneX)

    g.append('path')
      .datum(convergence)
      .attr('fill', 'none')
      .attr('stroke', '#a855f7')
      .attr('stroke-width', 2)
      .attr('d', line)

    // Dots
    g.selectAll('.dot')
      .data(convergence)
      .enter()
      .append('circle')
      .attr('cx', (_, i) => x(i))
      .attr('cy', d => y(d))
      .attr('r', 3)
      .attr('fill', '#a855f7')
      .attr('opacity', 0.7)

    // Running max line
    const runningMax: number[] = []
    let max = 0
    for (const v of convergence) {
      max = Math.max(max, v)
      runningMax.push(max)
    }

    g.append('path')
      .datum(runningMax)
      .attr('fill', 'none')
      .attr('stroke', '#f97316')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4,3')
      .attr('d', line)

  }, [convergence])

  return (
    <div className="p-4">
      <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
        Reward Dashboard
      </h2>

      <div className="flex gap-6">
        {/* Vertical gauge bars */}
        <div className="flex gap-4 items-end" style={{ minWidth: '340px' }}>
          {GAUGES.map(({ key, label, weight, color }) => {
            const value = node ? (node as any)[key] as number : 0
            return (
              <div key={key} className="flex flex-col items-center gap-1">
                <div className="relative bg-gray-800 rounded-full" style={{ width: '28px', height: '100px' }}>
                  <div
                    className="absolute bottom-0 rounded-full transition-all duration-300"
                    style={{
                      width: '28px',
                      height: `${value * 100}%`,
                      backgroundColor: color,
                      opacity: 0.8,
                    }}
                  />
                </div>
                <span className="text-[10px] font-mono text-gray-300">{value.toFixed(2)}</span>
                <span className="text-[9px] text-gray-500 text-center leading-tight" style={{ maxWidth: '60px' }}>
                  {label}
                </span>
                <span className="text-[8px] text-gray-600">w={weight}</span>
              </div>
            )
          })}
        </div>

        {/* Convergence chart */}
        <div className="flex-1" style={{ minHeight: '140px' }}>
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs text-gray-500">Composite Reward Over Iterations</span>
            <div className="flex gap-3 text-[10px] text-gray-500">
              <span className="flex items-center gap-1">
                <span className="w-3 h-0.5 bg-purple-500 inline-block" /> score
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-0.5 bg-orange-500 inline-block border-dashed" /> best
              </span>
            </div>
          </div>
          {convergence.length < 2 ? (
            <div className="h-32 flex items-center justify-center text-gray-600 text-xs">
              Convergence chart appears during discovery
            </div>
          ) : (
            <svg ref={chartRef} className="w-full" style={{ height: '130px' }} />
          )}
        </div>
      </div>
    </div>
  )
}
