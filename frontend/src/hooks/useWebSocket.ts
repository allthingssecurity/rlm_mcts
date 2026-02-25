import { useEffect, useRef, useCallback, useState } from "react";
import { WSEvent, TreeSnapshot } from "../types";

interface UseWebSocketReturn {
  connected: boolean;
  tree: TreeSnapshot;
  answer: string | null;
  confidence: number | null;
  searching: boolean;
  contextChars: number;
  sendQuestion: (question: string, videoIds: string[]) => void;
}

export function useWebSocket(): UseWebSocketReturn {
  const wsRef = useRef<WebSocket | null>(null);
  const retriesRef = useRef(0);
  const maxRetries = 5;

  const [connected, setConnected] = useState(false);
  const [tree, setTree] = useState<TreeSnapshot>({});
  const [answer, setAnswer] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [searching, setSearching] = useState(false);
  const [contextChars, setContextChars] = useState(0);

  const connect = useCallback(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      retriesRef.current = 0;
    };

    ws.onclose = () => {
      setConnected(false);
      wsRef.current = null;

      if (retriesRef.current < maxRetries) {
        const delay = Math.min(1000 * Math.pow(2, retriesRef.current), 10000);
        retriesRef.current += 1;
        setTimeout(connect, delay);
      }
    };

    ws.onerror = () => {
      ws.close();
    };

    ws.onmessage = (evt) => {
      try {
        const data: WSEvent = JSON.parse(evt.data);
        handleMessage(data);
      } catch {
        // ignore malformed messages
      }
    };
  }, []);

  const handleMessage = useCallback((data: WSEvent) => {
    switch (data.event) {
      case "node_update":
        // Tree snapshot includes all nodes with current visits/values
        setTree(data.tree_snapshot);
        break;

      case "search_started":
        setSearching(true);
        setAnswer(null);
        setConfidence(null);
        setTree({});
        setContextChars(data.context_chars);
        break;

      case "answer_ready":
        setAnswer(data.answer);
        setConfidence(data.confidence);
        break;

      case "search_complete":
        setSearching(false);
        setTree(data.tree);
        if (data.answer) setAnswer(data.answer);
        if (data.confidence != null) setConfidence(data.confidence);
        break;

      case "error":
        setSearching(false);
        console.error("WS error:", data.message);
        break;

      case "pong":
        break;
    }
  }, []);

  useEffect(() => {
    const timer = setTimeout(connect, 300);
    return () => {
      clearTimeout(timer);
      wsRef.current?.close();
    };
  }, [connect]);

  const sendQuestion = useCallback(
    (question: string, videoIds: string[]) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({
            type: "ask",
            question,
            video_ids: videoIds,
          })
        );
      }
    },
    []
  );

  return {
    connected,
    tree,
    answer,
    confidence,
    searching,
    contextChars,
    sendQuestion,
  };
}
