import { Download, GitBranch, Zap } from 'lucide-react';
import { useMemo } from 'react';

import { WorkflowGraph } from '../api/client';

type Props = {
  graph?: WorkflowGraph;
  theme: 'light' | 'dark';
  streaming?: boolean;
  onDownload?: (format: 'svg' | 'png') => void;
};

type Point = { x: number; y: number };

const levelForNode = (id: string) => {
  const map: Record<string, number> = {
    start: 0,
    initialization_agent: 1,
    assess_study_design: 2,
    parametric_assumptions_hold: 3,
    two_independent_groups: 3,
    chi_square_test: 3,
    fisher_exact_test: 3,
    paired_t_test: 4,
    wilcoxon_signed_rank_test: 4,
    independent_t_test: 4,
    nonparametric_tests: 4,
    summary: 5,
    reviewer_agent: 6,
    end: 7,
  };
  return map[id] ?? 4;
};

const layoutNodes = (nodes: WorkflowGraph['nodes']): Record<string, Point> => {
  const grouped: Record<number, WorkflowGraph['nodes']> = {};
  nodes.forEach((node) => {
    const lvl = levelForNode(node.id);
    grouped[lvl] = grouped[lvl] || [];
    grouped[lvl].push(node);
  });

  const positions: Record<string, Point> = {};
  const xSpacing = 180;
  const ySpacing = 90;
  Object.entries(grouped).forEach(([lvl, list]) => {
    const level = Number(lvl);
    const sorted = [...list].sort((a, b) => a.label.localeCompare(b.label));
    sorted.forEach((node, idx) => {
      positions[node.id] = { x: 60 + level * xSpacing, y: 60 + idx * ySpacing };
    });
  });
  return positions;
};

const colorForNode = (id: string, active?: string | null, visited?: Set<string>) => {
  const inPath = visited?.has(id);
  if (active && id === active) return { fill: '#f59e0b', stroke: '#f59e0b' };
  if (inPath) return { fill: '#0ea5e9', stroke: '#0ea5e9' };
  return { fill: '#0f172a', stroke: '#94a3b8' };
};

export function WorkflowGraphView({ graph, theme, streaming, onDownload }: Props) {
  const nodes = graph?.nodes || [];
  const edges = graph?.edges || [];
  const visited = useMemo(
    () => new Set(graph?.selected_path && graph.selected_path.length ? graph.selected_path : graph?.visited_nodes || []),
    [graph?.selected_path, graph?.visited_nodes]
  );
  const positions = useMemo(() => layoutNodes(nodes), [nodes]);
  const maxX = useMemo(() => Math.max(...Object.values(positions).map((p) => p.x), 320), [positions]);
  const maxY = useMemo(() => Math.max(...Object.values(positions).map((p) => p.y), 240), [positions]);

  if (!nodes.length) {
    return (
      <div
        className={`rounded-2xl border p-4 ${
          theme === 'dark' ? 'border-slate-800 bg-slate-900/70' : 'border-slate-200 bg-white'
        }`}
      >
        <div className="flex items-center gap-2 text-sm font-semibold text-slate-200">
          <GitBranch size={16} className="text-cyan-400" /> Workflow graph
        </div>
        <p className="mt-2 text-xs text-slate-400">Graph metadata not loaded yet.</p>
      </div>
    );
  }

  const activeLabel = nodes.find((n) => n.id === graph?.active_node)?.label;

  return (
    <div
      className={`rounded-2xl border p-4 shadow-lg ${
        theme === 'dark' ? 'border-slate-800 bg-slate-900/70' : 'border-slate-200 bg-white'
      }`}
    >
      <div className="mb-3 flex items-center justify-between">
        <div>
          <p className="text-[10px] uppercase tracking-[0.2em] text-slate-500">Workflow graph</p>
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-100">
            <GitBranch size={16} className="text-cyan-400" />
            {activeLabel ? `Active: ${activeLabel}` : 'Decision path'}
            {streaming && <Zap size={14} className="text-amber-400 animate-pulse" />}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[10px] uppercase text-slate-300">
            {visited.size} / {nodes.length} visited
          </span>
          {graph?.assets?.svg_base64 && onDownload && (
            <>
              <button
                className="rounded-full border border-slate-700 px-2 py-1 text-[11px] font-semibold text-slate-200 hover:border-cyan-500"
                onClick={() => onDownload('svg')}
              >
                <Download size={12} className="inline" /> SVG
              </button>
              <button
                className="rounded-full border border-slate-700 px-2 py-1 text-[11px] font-semibold text-slate-200 hover:border-cyan-500"
                onClick={() => onDownload('png')}
              >
                <Download size={12} className="inline" /> PNG
              </button>
            </>
          )}
        </div>
      </div>

      <div className="relative overflow-hidden rounded-xl border border-slate-800/60 bg-gradient-to-br from-slate-950 to-slate-900">
        <svg viewBox={`0 0 ${maxX + 140} ${maxY + 120}`} className="w-full">
          {edges.map((edge) => {
            const src = positions[edge.source];
            const tgt = positions[edge.target];
            if (!src || !tgt) return null;
            const inPath = visited.has(edge.source) && visited.has(edge.target);
            return (
              <line
                key={`${edge.source}-${edge.target}`}
                x1={src.x + 60}
                y1={src.y}
                x2={tgt.x - 20}
                y2={tgt.y}
                stroke={inPath ? '#0ea5e9' : '#475569'}
                strokeWidth={2}
                strokeDasharray={edge.kind === 'conditional' ? '6 4' : '0'}
                opacity={0.9}
              />
            );
          })}

          {nodes.map((node) => {
            const pos = positions[node.id];
            const { fill, stroke } = colorForNode(node.id, graph?.active_node, visited);
            return (
              <g key={node.id} transform={`translate(${pos.x},${pos.y})`}>
                <rect
                  x={-20}
                  y={-22}
                  width={130}
                  height={44}
                  rx={12}
                  fill={fill}
                  stroke={stroke}
                  strokeWidth={2}
                  className={graph?.active_node === node.id && streaming ? 'animate-pulse' : ''}
                  opacity={0.95}
                />
                <text
                  x={45}
                  y={0}
                  dominantBaseline="middle"
                  textAnchor="middle"
                  fill="#e2e8f0"
                  fontSize="11"
                  fontFamily="Inter, Arial, sans-serif"
                >
                  {node.label}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="mt-3 flex flex-wrap gap-2 text-[11px] text-slate-400">
        {(graph?.selected_path || graph?.visited_nodes || []).map((nodeId) => {
          const nodeLabel = nodes.find((n) => n.id === nodeId)?.label || nodeId;
          const isActive = graph?.active_node === nodeId;
          return (
            <span
              key={nodeId}
              className={`rounded-full px-3 py-1 ${
                isActive
                  ? 'bg-amber-500/20 text-amber-200'
                  : 'bg-cyan-500/10 text-cyan-200'
              }`}
            >
              {nodeLabel}
            </span>
          );
        })}
        {!graph?.visited_nodes?.length && <span className="text-slate-500">Waiting for first step…</span>}
      </div>
    </div>
  );
}
