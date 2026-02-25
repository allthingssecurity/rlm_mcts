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

export type WSEvent =
  | { event: "node_update"; node: ReasoningNode; tree_snapshot: TreeSnapshot }
  | { event: "answer_ready"; answer: string; confidence: number }
  | { event: "search_started"; question: string; context_chars: number }
  | { event: "search_complete"; answer: string; confidence: number; tree: TreeSnapshot }
  | { event: "error"; message: string }
  | { event: "pong" };
