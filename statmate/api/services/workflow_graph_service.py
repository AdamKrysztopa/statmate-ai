"""Render workflow graph metadata into SVG/PNG assets."""

from __future__ import annotations

import base64
from typing import Any

from statmate.workflow.graph_metadata import get_workflow_graph


def _level_for_node(node_id: str) -> int:
    """Assign a deterministic layout level for each node."""
    level_map = {
        'start': 0,
        'initialization_agent': 1,
        'assess_study_design': 2,
        'parametric_assumptions_hold': 3,
        'two_independent_groups': 3,
        'chi_square_test': 3,
        'fisher_exact_test': 3,
        'paired_t_test': 4,
        'wilcoxon_signed_rank_test': 4,
        'independent_t_test': 4,
        'nonparametric_tests': 4,
        'summary': 5,
        'reviewer_agent': 6,
        'end': 7,
    }
    return level_map.get(node_id, 4)


def _layout_nodes(nodes: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """Produce simple grid coordinates for each node."""
    levels: dict[int, list[dict[str, Any]]] = {}
    for node in nodes:
        levels.setdefault(_level_for_node(node['id']), []).append(node)

    positions: dict[str, tuple[float, float]] = {}
    x_spacing = 230
    y_spacing = 120
    for level, items in levels.items():
        items_sorted = sorted(items, key=lambda n: n['label'])
        for idx, node in enumerate(items_sorted):
            x = 120 + level * x_spacing
            y = 80 + idx * y_spacing
            positions[node['id']] = (x, y)
    return positions


def _node_style(node: dict[str, Any], path: set[str], active: str | None) -> tuple[str, str]:
    """Return fill and stroke colors for a node based on state."""
    base_fill = '#0b162e'
    base_stroke = '#334155'
    if node['id'] in path:
        base_fill = '#0ea5e9'
        base_stroke = '#0ea5e9'
    if active and node['id'] == active:
        base_fill = '#f59e0b'
        base_stroke = '#f59e0b'
    return base_fill, base_stroke


def _edge_color(source: str, target: str, selected: set[str]) -> str:
    """Color edges when both endpoints are in the selected path."""
    return '#0ea5e9' if source in selected and target in selected else '#475569'


def render_workflow_graph(graph_state: dict[str, Any]) -> dict[str, Any]:
    """Render the workflow graph with highlighted path to SVG/PNG base64."""
    graph = graph_state or get_workflow_graph()
    nodes = graph.get('nodes', [])
    edges = graph.get('edges', [])
    visited = graph_state.get('visited_nodes') if graph_state else []
    selected = set(visited or [])
    active_node = graph_state.get('active_node') if graph_state else None

    positions = _layout_nodes(nodes)
    width = 220 + max((x for x, _ in positions.values()), default=0)
    height = 200 + max((y for _, y in positions.values()), default=0)

    svg_parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#0ea5e9"/></marker></defs>',
        '<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#0b1223"/><stop offset="100%" stop-color="#0e1c33"/></linearGradient>',
        '<rect x="0" y="0" width="100%" height="100%" rx="18" fill="url(#bg)" stroke="#1f2937" stroke-width="2"/>',
    ]

    # Edges
    for edge in edges:
        source = edge['source']
        target = edge['target']
        if source not in positions or target not in positions:
            continue
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        color = _edge_color(source, target, selected)
        svg_parts.append(
            f'<line x1="{x1+90}" y1="{y1}" x2="{x2-50}" y2="{y2}" stroke="{color}" stroke-width="2.5" marker-end="url(#arrow)" opacity="0.85" />'
        )

    # Nodes
    for node in nodes:
        nid = node['id']
        label = node['label']
        x, y = positions.get(nid, (0, 0))
        fill, stroke = _node_style(node, selected, active_node)
        radius = 48 if node['kind'] in ('start', 'end') else 64
        svg_parts.append(
            f'<g><rect x="{x-28}" y="{y-36}" rx="14" ry="14" width="{radius+56}" height="72" fill="{fill}" stroke="{stroke}" stroke-width="2.5" opacity="0.96" />'
            f'<text x="{x+radius/2}" y="{y-4}" dominant-baseline="middle" text-anchor="middle" fill="#e2e8f0" font-family="Inter, Arial, sans-serif" font-size="13" font-weight="600">{label}</text></g>'
        )

    svg_parts.append('</svg>')
    svg = ''.join(svg_parts)
    svg_b64 = base64.b64encode(svg.encode('utf-8')).decode('ascii')

    png_b64: str | None = None
    try:
        import cairosvg  # type: ignore

        png_bytes = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
        png_b64 = base64.b64encode(png_bytes).decode('ascii')
    except Exception:
        png_b64 = None

    alt_text = f"Workflow graph highlighting {active_node or 'current path'}"
    payload = {
        'svg': svg,
        'svg_base64': svg_b64,
        'png_base64': png_b64,
        'alt': alt_text,
    }
    return payload
