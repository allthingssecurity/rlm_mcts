import { useEffect, useRef, useMemo } from 'react'
import * as d3 from 'd3'
import type { TreeSnapshot, RubricNode } from '../types'

interface Props {
  tree: TreeSnapshot | null
  selectedNodeId: string | null
  onSelectNode: (node: RubricNode) => void
}

const NODE_COLORS: Record<string, string> = {
  root: '#3b82f6',       // blue
  hypothesis: '#a855f7', // purple
  refinement: '#06b6d4', // cyan
  final: '#f97316',      // orange
}

interface TreeNode {
  data: RubricNode
  children: TreeNode[]
}

export default function MCTSTreeViz({ tree, selectedNodeId, onSelectNode }: Props) {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // Build hierarchical data from flat nodes
  const hierarchy = useMemo(() => {
    if (!tree) return null

    const nodes = tree.nodes
    const rootId = tree.root_id
    const rootNode = nodes[rootId]
    if (!rootNode) return null

    function buildTree(nodeId: string): TreeNode {
      const node = nodes[nodeId]
      return {
        data: node,
        children: (node.children_ids || [])
          .filter(id => nodes[id])
          .map(id => buildTree(id)),
      }
    }

    return buildTree(rootId)
  }, [tree])

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || !hierarchy) return

    const container = containerRef.current
    const width = container.clientWidth
    const height = container.clientHeight - 30 // leave room for title

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const margin = { top: 30, right: 30, bottom: 30, left: 60 }
    const innerW = width - margin.left - margin.right
    const innerH = height - margin.top - margin.bottom

    const g = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Create d3 hierarchy
    const root = d3.hierarchy<TreeNode>(hierarchy, d => d.children)
    const treeLayout = d3.tree<TreeNode>().size([innerH, innerW])
    treeLayout(root)

    // Draw links
    g.selectAll('.link')
      .data(root.links())
      .enter()
      .append('path')
      .attr('class', 'link')
      .attr('d', d3.linkHorizontal<d3.HierarchyPointLink<TreeNode>, d3.HierarchyPointNode<TreeNode>>()
        .x(d => d.y)
        .y(d => d.x) as any)
      .attr('fill', 'none')
      .attr('stroke', '#374151')
      .attr('stroke-width', 1.5)

    // Draw nodes
    const nodeG = g.selectAll('.node')
      .data(root.descendants())
      .enter()
      .append('g')
      .attr('class', 'node')
      .attr('transform', d => `translate(${d.y},${d.x})`)
      .style('cursor', 'pointer')
      .on('click', (_, d) => onSelectNode(d.data.data))

    // Node circles
    nodeG.append('circle')
      .attr('r', d => {
        const visits = d.data.data.visits || 1
        return Math.max(6, Math.min(20, 4 + visits * 2))
      })
      .attr('fill', d => {
        const color = NODE_COLORS[d.data.data.node_type] || '#6b7280'
        return color
      })
      .attr('opacity', d => {
        const composite = d.data.data.reward_composite
        return Math.max(0.3, composite)
      })
      .attr('stroke', d => d.data.data.id === selectedNodeId ? '#fff' : 'transparent')
      .attr('stroke-width', 2)

    // Best node glow
    if (tree?.best_node_id) {
      nodeG.filter(d => d.data.data.id === tree.best_node_id)
        .append('circle')
        .attr('r', d => Math.max(10, Math.min(24, 8 + (d.data.data.visits || 1) * 2)))
        .attr('fill', 'none')
        .attr('stroke', '#f97316')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '3,3')
        .attr('opacity', 0.6)
    }

    // Labels
    nodeG.append('text')
      .attr('dy', -12)
      .attr('text-anchor', 'middle')
      .attr('fill', '#9ca3af')
      .attr('font-size', '9px')
      .text(d => {
        if (d.data.data.node_type === 'root') return 'ROOT'
        return `${d.data.data.reward_composite.toFixed(2)}`
      })

  }, [hierarchy, selectedNodeId, tree, onSelectNode])

  return (
    <div ref={containerRef} className="h-full flex flex-col">
      <div className="px-4 py-2 border-b border-gray-800 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">MCTS Tree</h2>
        {tree && (
          <span className="text-xs text-gray-500">
            {Object.keys(tree.nodes).length} nodes
          </span>
        )}
      </div>
      <div className="flex-1 overflow-hidden">
        {!tree ? (
          <div className="h-full flex items-center justify-center text-gray-600 text-sm">
            Load dataset and start discovery to see the tree
          </div>
        ) : (
          <svg ref={svgRef} className="w-full h-full" />
        )}
      </div>
      {/* Legend */}
      <div className="px-4 py-1.5 border-t border-gray-800 flex gap-4 text-[10px] text-gray-500">
        {Object.entries(NODE_COLORS).map(([type, color]) => (
          <span key={type} className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full inline-block" style={{ backgroundColor: color }} />
            {type}
          </span>
        ))}
      </div>
    </div>
  )
}
