export interface TestResult {
  predicted: number
  actual: number
  error?: number
}

export interface RewardBreakdown {
  generalization: number
  calibration: number
  discrimination: number
  validity: number
  iteration: number
  composite: number
  weights?: Record<string, number>
}

export interface RubricNode {
  id: string
  rubric_code: string
  node_type: 'root' | 'hypothesis' | 'refinement' | 'final'
  depth: number
  visits: number
  reward_generalization: number
  reward_calibration: number
  reward_discrimination: number
  reward_validity: number
  reward_iteration: number
  reward_composite: number
  train_mae: number
  eval_mae: number
  stdout: string
  stderr: string
  execution_success: boolean
  parent_id: string | null
  children_ids: string[]
  train_results: TestResult[]
  eval_results: TestResult[]
}

export interface TreeSnapshot {
  root_id: string
  nodes: Record<string, RubricNode>
  best_node_id: string | null
}

export interface DatasetInfo {
  num_training: number
  num_eval: number
  train_score_mean: number
  train_score_min: number
  train_score_max: number
  eval_score_mean: number
  score_distribution: Record<string, number>
}

export interface EvalResult {
  best_rubric_code: string
  eval_mae: number
  eval_accuracy: number
  eval_count: number
  eval_results: TestResult[]
  best_composite: number
}

export type WSEvent =
  | { event: 'discovery_started'; num_training: number; num_eval: number }
  | { event: 'node_update'; node: RubricNode; tree_snapshot: TreeSnapshot; iteration: number; total_iterations: number }
  | { event: 'discovery_complete'; best_rubric_code: string; best_score: number; eval_results: EvalResult; tree_snapshot: TreeSnapshot }
  | { event: 'error'; message: string }
  | { event: 'pong' }
