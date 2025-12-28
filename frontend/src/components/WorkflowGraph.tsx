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
    design_verification: 2,
    design_reconciliation: 2,
    parametric_assumptions_hold: 3,
    two_independent_groups: 3,
    anova_assumptions: 3,
    chi_square_test: 3,
    fisher_exact_test: 3,
    paired_t_test: 4,
    wilcoxon_signed_rank_test: 4,
    independent_t_test: 4,
    nonparametric_tests: 4,
    one_way_anova: 4,
    kruskal_wallis_h_test: 4,
    anova_repeated_measures: 4,
    friedman_test: 4,
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
  const xSpacing = 240;
  const ySpacing = 110;
  Object.entries(grouped).forEach(([lvl, list]) => {
    const level = Number(lvl);
    const sorted = [...list].sort((a, b) => a.label.localeCompare(b.label));
    sorted.forEach((node, idx) => {
      positions[node.id] = { x: 80 + level * xSpacing, y: 90 + idx * ySpacing };
    });
  });
  return positions;
};

const colorForNode = (id: string, active?: string | null, visited?: Set<string>) => {
  const inPath = visited?.has(id);
  if (active && id === active) return { fill: '#f59e0b', stroke: '#f59e0b' };
  if (inPath) return { fill: '#0ea5e9', stroke: '#0ea5e9' };
  return { fill: '#0b1c34', stroke: '#334155' };
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
  const visitedCount = visited.size || 0;

  return (
    <div
      className={`relative rounded-3xl border p-5 shadow-2xl ${
        theme === 'dark' ? 'border-slate-800 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950' : 'border-slate-200 bg-white'
      }`}
    >
      <div className="mb-4 flex items-center justify-between">
        <div>
          <p className="text-[10px] uppercase tracking-[0.24em] text-slate-500">Workflow graph</p>
          <div className="flex items-center gap-3 text-lg font-bold text-slate-100">
            <GitBranch size={18} className="text-cyan-400" />
            <span>{activeLabel ? `Active: ${activeLabel}` : 'Decision path'}</span>
            {streaming && <Zap size={16} className="text-amber-400 animate-pulse" />}
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="rounded-full bg-slate-800 px-4 py-2 text-xs font-semibold text-slate-200">
            {visitedCount} / {nodes.length} visited
          </div>
          {graph?.assets?.svg_base64 && onDownload && (
            <>
              <button
                className="rounded-full border border-slate-700/70 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:border-cyan-500"
                onClick={() => onDownload('svg')}
              >
                <Download size={14} className="inline" /> SVG
              </button>
              <button
                className="rounded-full border border-slate-700/70 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:border-cyan-500"
                onClick={() => onDownload('png')}
              >
                <Download size={14} className="inline" /> PNG
              </button>
            </>
          )}
        </div>
      </div>

      <div className="relative overflow-hidden rounded-2xl border border-slate-800/60 bg-slate-950">
        <svg viewBox={`0 0 ${maxX + 320} ${maxY + 260}`} className="w-full">
          <defs>
            <linearGradient id="edgeGlow" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#0ea5e9" stopOpacity="0.6" />
              <stop offset="100%" stopColor="#0ea5e9" stopOpacity="0.2" />
            </linearGradient>
          </defs>
          {edges.map((edge) => {
            const src = positions[edge.source];
            const tgt = positions[edge.target];
            if (!src || !tgt) return null;
            const inPath = visited.has(edge.source) && visited.has(edge.target);
            return (
              <line
                key={`${edge.source}-${edge.target}`}
                x1={src.x + 80}
                y1={src.y}
                x2={tgt.x - 40}
                y2={tgt.y}
                stroke={inPath ? 'url(#edgeGlow)' : '#475569'}
                strokeWidth={inPath ? 3 : 1.6}
                strokeDasharray={edge.kind === 'conditional' ? '7 5' : '0'}
                opacity={0.95}
              />
            );
          })}

          {nodes.map((node) => {
            const pos = positions[node.id];
            const { fill, stroke } = colorForNode(node.id, graph?.active_node, visited);
            return (
              <g key={node.id} transform={`translate(${pos.x},${pos.y})`}>
                <rect
                  x={-28}
                  y={-30}
                  width={160}
                  height={64}
                  rx={16}
                  fill={fill}
                  stroke={stroke}
                  strokeWidth={2.4}
                  className={graph?.active_node === node.id && streaming ? 'animate-pulse' : ''}
                  opacity={0.97}
                />
                <text
                  x={52}
                  y={0}
                  dominantBaseline="middle"
                  textAnchor="middle"
                  fill="#e2e8f0"
                  fontSize="12"
                  fontWeight={600}
                  fontFamily="Inter, Arial, sans-serif"
                >
                  {node.label}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="mt-4 flex flex-wrap gap-3 text-xs">
        {(graph?.selected_path || graph?.visited_nodes || []).map((nodeId) => {
          const nodeLabel = nodes.find((n) => n.id === nodeId)?.label || nodeId;
          const isActive = graph?.active_node === nodeId;
          return (
            <span
              key={nodeId}
              className={`rounded-full px-4 py-2 font-semibold ${
                isActive ? 'bg-amber-500/20 text-amber-200' : 'bg-cyan-500/15 text-cyan-100'
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
