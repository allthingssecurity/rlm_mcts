export interface ReasoningNode {
  id: string;
  content: string;
  node_type: "question" | "strategy" | "code" | "result" | "answer";
  parent_id: string | null;
  children: string[];
  visits: number;
  total_value: number;
  avg_value: number;
  depth: number;
  code: string;
  repl_stdout: string;
  repl_stderr: string;
  repl_vars: Record<string, string>;
  execution_ms: number;
}

export type TreeSnapshot = Record<string, ReasoningNode>;

export interface VideoInfo {
  video_id: string;
  title: string;
  duration: number;
  channel: string;
  segment_count?: number;
  transcript_chars?: number;
  transcript_preview?: string;
  error?: string;
}

export interface PlainRLMStep {
  step_number: number;
  code: string;
  stdout: string;
  stderr: string;
  execution_ms: number;
  success: boolean;
}

export interface ComparisonMetrics {
  total_time_ms: number;
  llm_calls: number;
  code_executions: number;
  successful_code_blocks: number;
  answer_length: number;
  confidence: number;
  unique_strategies?: number;
  max_depth_reached?: number;
  avg_node_value?: number;
}

export interface ComparisonResult {
  plain: {
    answer: string;
    confidence: number;
    metrics: ComparisonMetrics;
    steps: PlainRLMStep[];
  };
  mcts: {
    answer: string;
    confidence: number;
    metrics: ComparisonMetrics;
    tree: TreeSnapshot;
  };
}

export type WSEvent =
  | { event: "node_update"; node: ReasoningNode; tree_snapshot: TreeSnapshot }
  | { event: "answer_ready"; answer: string; confidence: number }
  | { event: "search_started"; question: string; context_chars: number }
  | { event: "search_complete"; answer: string; confidence: number; tree: TreeSnapshot }
  | { event: "plain_step"; step: PlainRLMStep }
  | { event: "comparison_complete"; plain: ComparisonResult["plain"]; mcts: ComparisonResult["mcts"] }
  | { event: "error"; message: string }
  | { event: "pong" };
