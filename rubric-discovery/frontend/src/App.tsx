import { useWebSocket } from './hooks/useWebSocket'
import DatasetPanel from './components/DatasetPanel'
import MCTSTreeViz from './components/MCTSTreeViz'
import NodeDetail from './components/NodeDetail'
import RewardDashboard from './components/RewardDashboard'
import EvalResults from './components/EvalResults'

export default function App() {
  const {
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
  } = useWebSocket()

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-3 flex items-center gap-4">
        <h1 className="text-lg font-bold tracking-tight">
          Rubric Discovery
          <span className="text-purple-400 ml-2 font-normal text-sm">MCTS Agent</span>
        </h1>
        {discovering && (
          <span className="text-xs text-cyan-400 animate-pulse">
            Iteration {iteration}/{totalIterations}
          </span>
        )}
        {error && (
          <span className="text-xs text-red-400 ml-auto truncate max-w-md">
            {error}
          </span>
        )}
      </header>

      {/* Main 3-panel layout */}
      <div className="flex-1 grid grid-cols-12 gap-0.5 p-0.5 overflow-hidden" style={{ height: 'calc(100vh - 52px)' }}>
        {/* Left: Dataset Panel */}
        <div className="col-span-3 bg-gray-900 rounded-lg overflow-y-auto">
          <DatasetPanel
            datasetInfo={datasetInfo}
            discovering={discovering}
            iteration={iteration}
            totalIterations={totalIterations}
            onLoadDataset={loadDataset}
            onStartDiscovery={startDiscovery}
          />
        </div>

        {/* Center: MCTS Tree Visualization */}
        <div className="col-span-5 bg-gray-900 rounded-lg overflow-hidden">
          <MCTSTreeViz
            tree={tree}
            selectedNodeId={selectedNode?.id ?? null}
            onSelectNode={selectNode}
          />
        </div>

        {/* Right: Node Detail */}
        <div className="col-span-4 bg-gray-900 rounded-lg overflow-y-auto">
          <NodeDetail node={selectedNode} />
        </div>
      </div>

      {/* Bottom: Reward Dashboard */}
      <div className="bg-gray-900 mx-0.5 mb-0.5 rounded-lg" style={{ minHeight: '180px' }}>
        <RewardDashboard
          node={selectedNode}
          convergence={convergence}
        />
      </div>

      {/* Eval Results (shown after discovery_complete) */}
      {evalResult && (
        <div className="bg-gray-900 mx-0.5 mb-0.5 rounded-lg">
          <EvalResults result={evalResult} />
        </div>
      )}
    </div>
  )
}
