import { useState, useCallback, useRef, useEffect } from 'react'
import type { TreeSnapshot, RubricNode, DatasetInfo, EvalResult, WSEvent } from '../types'

export function useWebSocket() {
  const [tree, setTree] = useState<TreeSnapshot | null>(null)
  const [selectedNode, setSelectedNode] = useState<RubricNode | null>(null)
  const [discovering, setDiscovering] = useState(false)
  const [iteration, setIteration] = useState(0)
  const [totalIterations, setTotalIterations] = useState(15)
  const [evalResult, setEvalResult] = useState<EvalResult | null>(null)
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [convergence, setConvergence] = useState<number[]>([])

  const wsRef = useRef<WebSocket | null>(null)
  const pingRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws`)

    ws.onopen = () => {
      // Heartbeat
      pingRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'ping' }))
        }
      }, 30000)
    }

    ws.onmessage = (evt) => {
      const data: WSEvent = JSON.parse(evt.data)

      switch (data.event) {
        case 'discovery_started':
          setDiscovering(true)
          setIteration(0)
          setEvalResult(null)
          setConvergence([])
          setError(null)
          break

        case 'node_update':
          setTree(data.tree_snapshot)
          setIteration(data.iteration)
          setTotalIterations(data.total_iterations)
          setSelectedNode(data.node)
          setConvergence(prev => [...prev, data.node.reward_composite])
          break

        case 'discovery_complete':
          setDiscovering(false)
          setTree(data.tree_snapshot)
          setEvalResult(data.eval_results)
          break

        case 'error':
          setError(data.message)
          setDiscovering(false)
          break

        case 'pong':
          break
      }
    }

    ws.onclose = () => {
      if (pingRef.current) clearInterval(pingRef.current)
    }

    wsRef.current = ws
  }, [])

  useEffect(() => {
    connect()
    return () => {
      if (pingRef.current) clearInterval(pingRef.current)
      wsRef.current?.close()
    }
  }, [connect])

  const loadDataset = useCallback(async () => {
    try {
      const res = await fetch('/api/load-dataset', { method: 'POST' })
      const info: DatasetInfo = await res.json()
      setDatasetInfo(info)
      return info
    } catch (e) {
      setError(`Failed to load dataset: ${e}`)
      return null
    }
  }, [])

  const startDiscovery = useCallback((maxIterations = 15, maxDepth = 4) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      connect()
      // Retry after connection
      setTimeout(() => startDiscovery(maxIterations, maxDepth), 500)
      return
    }
    setConvergence([])
    setEvalResult(null)
    setError(null)
    wsRef.current.send(JSON.stringify({
      type: 'discover',
      max_iterations: maxIterations,
      max_depth: maxDepth,
    }))
  }, [connect])

  const selectNode = useCallback((node: RubricNode | null) => {
    setSelectedNode(node)
  }, [])

  return {
    tree,
    selectedNode,
    discovering,
    iteration,
    totalIterations,
    evalResult,
    datasetInfo,
    error,
    convergence,
    loadDataset,
    startDiscovery,
    selectNode,
  }
}
