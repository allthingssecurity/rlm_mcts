import React, { useEffect, useRef, useCallback } from "react";
import * as d3 from "d3";
import { TreeSnapshot, ReasoningNode } from "../types";

interface Props {
  tree: TreeSnapshot;
  onNodeClick: (node: ReasoningNode) => void;
}

interface HierNode {
  id: string;
  content: string;
  node_type: string;
  visits: number;
  avg_value: number;
  children?: HierNode[];
}

const NODE_COLORS: Record<string, string> = {
  question: "#3b82f6",   // blue
  strategy: "#a855f7",   // purple
  code: "#06b6d4",       // cyan
  result: "#22c55e",     // green
  answer: "#f97316",     // orange
};

const WIDTH = 900;
const HEIGHT = 500;
const MARGIN = { top: 40, right: 40, bottom: 40, left: 40 };

export default function MCTSTreeViz({ tree, onNodeClick }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const previousNodeCount = useRef(0);

  const toHierarchy = useCallback(
    (snapshot: TreeSnapshot): HierNode | null => {
      const nodes = Object.values(snapshot);
      if (nodes.length === 0) return null;

      const root = nodes.find((n) => n.parent_id === null);
      if (!root) return null;

      const buildTree = (node: ReasoningNode): HierNode => {
        const children = node.children
          .map((cid) => snapshot[cid])
          .filter(Boolean)
          .map(buildTree);

        return {
          id: node.id,
          content: node.content,
          node_type: node.node_type,
          visits: node.visits,
          avg_value: node.avg_value,
          children: children.length > 0 ? children : undefined,
        };
      };

      return buildTree(root);
    },
    []
  );

  useEffect(() => {
    if (!svgRef.current) return;

    const hierData = toHierarchy(tree);
    if (!hierData) {
      d3.select(svgRef.current).selectAll("*").remove();
      previousNodeCount.current = 0;
      return;
    }

    const svg = d3.select(svgRef.current);

    const treeLayout = d3.tree<HierNode>().size([
      WIDTH - MARGIN.left - MARGIN.right,
      HEIGHT - MARGIN.top - MARGIN.bottom,
    ]);

    const rootHier = d3.hierarchy(hierData);
    treeLayout(rootHier);

    const descendants = rootHier.descendants();
    const links = rootHier.links();
    const isNew = descendants.length > previousNodeCount.current;
    previousNodeCount.current = descendants.length;

    let g = svg.select<SVGGElement>("g.tree-container");
    if (g.empty()) {
      g = svg
        .append("g")
        .attr("class", "tree-container")
        .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);
    }

    // --- LINKS ---
    const linkSel = g
      .selectAll<SVGPathElement, d3.HierarchyLink<HierNode>>("path.link")
      .data(links, (d) => `${(d.source.data as HierNode).id}-${(d.target.data as HierNode).id}`);

    linkSel.exit().transition().duration(300).style("opacity", 0).remove();

    const linkEnter = linkSel
      .enter()
      .append("path")
      .attr("class", "link")
      .attr("fill", "none")
      .attr("stroke", "#4b5563")
      .attr("stroke-width", 1.5)
      .style("opacity", 0);

    linkEnter
      .merge(linkSel)
      .transition()
      .duration(400)
      .style("opacity", 1)
      .attr("d", (d) => {
        const sx = d.source.x!;
        const sy = d.source.y!;
        const tx = d.target.x!;
        const ty = d.target.y!;
        return `M${sx},${sy} C${sx},${(sy + ty) / 2} ${tx},${(sy + ty) / 2} ${tx},${ty}`;
      })
      .attr("stroke", "#4b5563")
      .attr("stroke-width", 1.5);

    // --- NODES ---
    const nodeSel = g
      .selectAll<SVGGElement, d3.HierarchyPointNode<HierNode>>("g.node")
      .data(descendants, (d) => (d.data as HierNode).id);

    nodeSel.exit().transition().duration(300).style("opacity", 0).remove();

    const nodeEnter = nodeSel
      .enter()
      .append("g")
      .attr("class", () => `node ${isNew ? "new" : ""}`)
      .attr("transform", (d) => `translate(${d.x},${d.y})`)
      .style("cursor", "pointer")
      .on("click", (_event, d) => {
        const nodeData = tree[d.data.id];
        if (nodeData) onNodeClick(nodeData);
      });

    nodeEnter
      .append("circle")
      .attr("r", 0)
      .transition()
      .duration(500)
      .ease(d3.easeBounceOut)
      .attr("r", (d) => Math.max(6, Math.min(16, 4 + d.data.visits * 2)));

    nodeEnter
      .append("text")
      .attr("dy", -16)
      .attr("text-anchor", "middle")
      .attr("fill", "#d1d5db")
      .attr("font-size", "10px")
      .text((d) => {
        const c = d.data.content;
        return c.length > 30 ? c.slice(0, 28) + "..." : c;
      });

    nodeEnter
      .append("text")
      .attr("dy", 24)
      .attr("text-anchor", "middle")
      .attr("fill", "#9ca3af")
      .attr("font-size", "9px")
      .attr("class", "value-label");

    const nodeMerge = nodeEnter.merge(nodeSel);

    nodeMerge
      .transition()
      .duration(400)
      .attr("transform", (d) => `translate(${d.x},${d.y})`);

    nodeMerge
      .select("circle")
      .transition()
      .duration(400)
      .attr("r", (d) => Math.max(6, Math.min(16, 4 + d.data.visits * 2)))
      .attr("fill", (d) => NODE_COLORS[d.data.node_type] || "#6b7280")
      .attr("fill-opacity", (d) => 0.3 + d.data.avg_value * 0.7)
      .attr("stroke", (d) => NODE_COLORS[d.data.node_type] || "#6b7280")
      .attr("stroke-width", 2);

    // Update labels with current values
    nodeMerge.select("text:first-of-type").text((d) => {
      const c = d.data.content;
      return c.length > 30 ? c.slice(0, 28) + "..." : c;
    });

    nodeMerge.select(".value-label").text((d) => {
      if (d.data.visits === 0) return "";
      return `v:${d.data.avg_value.toFixed(2)} n:${d.data.visits}`;
    });
  }, [tree, toHierarchy, onNodeClick]);

  const nodeCount = Object.keys(tree).length;

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-800">
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">
          RLM + MCTS Reasoning Tree
        </h2>
        <div className="flex items-center gap-4 text-xs text-gray-500">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-blue-500 inline-block" />
            Question
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-purple-500 inline-block" />
            Strategy
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-cyan-500 inline-block" />
            Code
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-orange-500 inline-block" />
            Answer
          </span>
          {nodeCount > 0 && (
            <span className="font-mono">{nodeCount} nodes</span>
          )}
        </div>
      </div>
      <svg
        ref={svgRef}
        className="mcts-tree"
        width={WIDTH}
        height={HEIGHT}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        style={{ maxWidth: "100%", height: "auto" }}
      />
    </div>
  );
}
