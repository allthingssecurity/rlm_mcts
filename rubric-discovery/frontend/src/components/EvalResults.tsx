import { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import type { EvalResult } from '../types'

interface Props {
  result: EvalResult
}

export default function EvalResults({ result }: Props) {
  const scatterRef = useRef<SVGSVGElement>(null)

  // Scatter plot: predicted vs actual
  useEffect(() => {
    if (!scatterRef.current || !result.eval_results.length) return

    const svg = d3.select(scatterRef.current)
    svg.selectAll('*').remove()

    const width = scatterRef.current.clientWidth
    const height = 250
    const margin = { top: 15, right: 15, bottom: 35, left: 40 }
    const innerW = width - margin.left - margin.right
    const innerH = height - margin.top - margin.bottom

    svg.attr('height', height)
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    const x = d3.scaleLinear().domain([0, 1]).range([0, innerW])
    const y = d3.scaleLinear().domain([0, 1]).range([innerH, 0])

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).ticks(5))
      .call(g => g.selectAll('text').attr('fill', '#9ca3af').attr('font-size', '10px'))
      .call(g => g.selectAll('line').attr('stroke', '#374151'))
      .call(g => g.select('.domain').attr('stroke', '#374151'))

    g.append('g')
      .call(d3.axisLeft(y).ticks(5))
      .call(g => g.selectAll('text').attr('fill', '#9ca3af').attr('font-size', '10px'))
      .call(g => g.selectAll('line').attr('stroke', '#374151'))
      .call(g => g.select('.domain').attr('stroke', '#374151'))

    // Axis labels
    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 30)
      .attr('text-anchor', 'middle')
      .attr('fill', '#6b7280')
      .attr('font-size', '10px')
      .text('Actual Score')

    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerH / 2)
      .attr('y', -30)
      .attr('text-anchor', 'middle')
      .attr('fill', '#6b7280')
      .attr('font-size', '10px')
      .text('Predicted Score')

    // Perfect diagonal
    g.append('line')
      .attr('x1', 0).attr('y1', innerH)
      .attr('x2', innerW).attr('y2', 0)
      .attr('stroke', '#374151')
      .attr('stroke-dasharray', '4,4')

    // Data points
    g.selectAll('.point')
      .data(result.eval_results)
      .enter()
      .append('circle')
      .attr('cx', d => x(d.actual))
      .attr('cy', d => y(d.predicted))
      .attr('r', 4)
      .attr('fill', d => {
        const err = Math.abs(d.predicted - d.actual)
        if (err < 0.15) return '#22c55e'
        if (err < 0.3) return '#eab308'
        return '#ef4444'
      })
      .attr('opacity', 0.7)

  }, [result])

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-green-400 uppercase tracking-wider">
          Discovery Complete
        </h2>
        <span className="text-xs text-gray-500">
          Evaluated on {result.eval_count} held-out examples
        </span>
      </div>

      <div className="grid grid-cols-12 gap-4">
        {/* Metrics */}
        <div className="col-span-3 space-y-2">
          <MetricCard label="Eval MAE" value={result.eval_mae.toFixed(4)} color="text-yellow-400" />
          <MetricCard label="Generalization Accuracy" value={`${(result.eval_accuracy * 100).toFixed(1)}%`} color="text-green-400" />
          <MetricCard label="Best Composite" value={result.best_composite.toFixed(4)} color="text-purple-400" />
        </div>

        {/* Scatter plot */}
        <div className="col-span-4">
          <h3 className="text-xs text-gray-500 mb-1">Predicted vs Actual (Eval Set)</h3>
          <svg ref={scatterRef} className="w-full bg-gray-950 rounded border border-gray-800" />
          <div className="flex gap-3 mt-1 text-[10px] text-gray-500 justify-center">
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-green-500 inline-block" /> &lt;0.15</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-yellow-500 inline-block" /> &lt;0.30</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-500 inline-block" /> &gt;0.30</span>
          </div>
        </div>

        {/* Best rubric code */}
        <div className="col-span-5">
          <h3 className="text-xs text-gray-500 mb-1">Best Discovered Rubric</h3>
          <pre className="bg-gray-950 border border-gray-800 rounded p-3 text-xs text-green-300 overflow-auto max-h-64 leading-relaxed">
            {result.best_rubric_code}
          </pre>
        </div>
      </div>

      {/* Per-example table */}
      {result.eval_results.length > 0 && (
        <div>
          <h3 className="text-xs text-gray-500 uppercase mb-1">
            Per-Example Eval Results (top 30)
          </h3>
          <div className="max-h-48 overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-gray-900">
                <tr className="text-gray-500">
                  <th className="text-left py-1 pr-2">#</th>
                  <th className="text-right py-1 pr-2">Predicted</th>
                  <th className="text-right py-1 pr-2">Actual</th>
                  <th className="text-right py-1">Error</th>
                </tr>
              </thead>
              <tbody>
                {result.eval_results.slice(0, 30).map((r, i) => (
                  <tr key={i} className="border-t border-gray-800">
                    <td className="py-0.5 pr-2 text-gray-600">{i + 1}</td>
                    <td className="py-0.5 pr-2 text-right font-mono">{r.predicted.toFixed(3)}</td>
                    <td className="py-0.5 pr-2 text-right font-mono">{r.actual.toFixed(3)}</td>
                    <td className={`py-0.5 text-right font-mono ${
                      Math.abs(r.predicted - r.actual) > 0.3 ? 'text-red-400' : 'text-gray-400'
                    }`}>
                      {Math.abs(r.predicted - r.actual).toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

function MetricCard({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="bg-gray-800 rounded p-3">
      <div className="text-[10px] text-gray-500 uppercase">{label}</div>
      <div className={`text-xl font-mono ${color}`}>{value}</div>
    </div>
  )
}
