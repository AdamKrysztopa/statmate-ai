"""Canonical workflow graph metadata and helpers."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from statmate.core.config import NodeName


def normalize_node_id(name: str) -> str:
    """Convert a human-readable node label into a stable identifier."""
    slug = re.sub(r'[^a-z0-9]+', '_', name.strip().lower())
    slug = re.sub(r'_+', '_', slug).strip('_')
    return slug or 'node'


@dataclass(frozen=True)
class WorkflowNode:
    """Representation of a workflow node."""

    id: str
    label: str
    kind: str
    transitions: list[str]


@dataclass(frozen=True)
class WorkflowEdge:
    """Representation of a workflow edge."""

    source: str
    target: str
    kind: str = 'flow'


_NODE_ALIAS: dict[str, str] = {
    'initialization': NodeName.INITIALIZATION,
    'initialization agent': NodeName.INITIALIZATION,
    'initialization_agent': NodeName.INITIALIZATION,
    't-test': NodeName.PAIRED_T,
    'paired t-test': NodeName.PAIRED_T,
    'paired_t-test': NodeName.PAIRED_T,
    'assess study design': NodeName.ASSESS_STUDY_DESIGN,
    'assess_study_design': NodeName.ASSESS_STUDY_DESIGN,
    'normality of difference test': NodeName.NORMALITY_OF_DIFFERENCE,
    'normality of difference': NodeName.NORMALITY_OF_DIFFERENCE,
    'parametric assumptions hold?': NodeName.NORMALITY_OF_DIFFERENCE,
    'parametric_assumptions_hold?': NodeName.NORMALITY_OF_DIFFERENCE,
    'parametric assumptions hold': NodeName.NORMALITY_OF_DIFFERENCE,
    'two independent groups?': NodeName.TWO_INDEPENDENT_GROUPS,
    'two_independent_groups?': NodeName.TWO_INDEPENDENT_GROUPS,
    'paired t-test': NodeName.PAIRED_T,
    'paired_t_test': NodeName.PAIRED_T,
    'wilcoxon signed-rank test': NodeName.WILCOXON,
    'wilcoxon signed rank test': NodeName.WILCOXON,
    'wilcoxon': NodeName.WILCOXON,
    'independent t-test': NodeName.INDEP_T,
    'independent t test': NodeName.INDEP_T,
    "levene's test": NodeName.LEVENE,
    'one-way anova': NodeName.ANOVA_ONE_WAY,
    'anova assumptions': NodeName.ANOVA_ASSUMPTIONS,
    'kruskal-wallis': NodeName.KRUSKAL_WALLIS,
    'kruskal wallis': NodeName.KRUSKAL_WALLIS,
    'friedman test': NodeName.FRIEDMAN,
    'shapiro-wilk test': NodeName.SHAPIRO,
    'nonparametric tests': NodeName.NONPARAMETRIC,
    'nonparametric': NodeName.NONPARAMETRIC,
    'chi-square test': NodeName.CHI2,
    'chi square test': NodeName.CHI2,
    'fisher exact test': NodeName.FISHER,
    'summary': NodeName.SUMMARY,
    'reviewer': NodeName.REVIEWER,
    'reviewer agent': NodeName.REVIEWER,
    'design verification': NodeName.DESIGN_VERIFICATION,
    'design reconciliation': NodeName.DESIGN_RECONCILIATION,
    'descriptive summary': NodeName.DESCRIPTIVE_SUMMARY,
    'user intervention needed': NodeName.USER_INTERVENTION,
    'choice node': NodeName.CHOICE,
    'mcnemar test': NodeName.MCNEMAR,
    "welch's t-test": NodeName.WELCH,
    'welch t-test': NodeName.WELCH,
    'mann-whitney u test': NodeName.MANN,
    'cox regression': NodeName.COX_REGRESSION,
}


_NODES: list[WorkflowNode] = [
    WorkflowNode(id='start', label='Start', kind='start', transitions=[normalize_node_id(NodeName.INITIALIZATION)]),
    WorkflowNode(
        id=normalize_node_id(NodeName.INITIALIZATION),
        label=NodeName.INITIALIZATION,
        kind='agent',
        transitions=[
            normalize_node_id(NodeName.DESIGN_VERIFICATION),
        ],
    ),
    WorkflowNode(
        id=normalize_node_id(NodeName.DESIGN_VERIFICATION),
        label=NodeName.DESIGN_VERIFICATION,
        kind='checkpoint',
        transitions=[
            normalize_node_id(NodeName.ASSESS_STUDY_DESIGN),
            normalize_node_id(NodeName.CHI2),
            normalize_node_id(NodeName.FISHER),
            normalize_node_id(NodeName.NONPARAMETRIC),
            normalize_node_id(NodeName.ANOVA_ASSUMPTIONS),
            normalize_node_id(NodeName.ANOVA_ONE_WAY),
            normalize_node_id(NodeName.KRUSKAL_WALLIS),
            normalize_node_id(NodeName.ANOVA_RM),
            normalize_node_id(NodeName.FRIEDMAN),
            normalize_node_id(NodeName.MCNEMAR),
            normalize_node_id(NodeName.CHOICE),
            normalize_node_id(NodeName.COX_REGRESSION),
            normalize_node_id(NodeName.DESCRIPTIVE_SUMMARY),
            normalize_node_id(NodeName.USER_INTERVENTION),
            normalize_node_id(NodeName.DESIGN_RECONCILIATION),
        ],
    ),
    WorkflowNode(
        id=normalize_node_id(NodeName.DESIGN_RECONCILIATION),
        label=NodeName.DESIGN_RECONCILIATION,
        kind='checkpoint',
        transitions=[
            normalize_node_id(NodeName.ASSESS_STUDY_DESIGN),
            normalize_node_id(NodeName.CHI2),
            normalize_node_id(NodeName.FISHER),
            normalize_node_id(NodeName.NONPARAMETRIC),
            normalize_node_id(NodeName.ANOVA_ASSUMPTIONS),
            normalize_node_id(NodeName.ANOVA_ONE_WAY),
            normalize_node_id(NodeName.KRUSKAL_WALLIS),
            normalize_node_id(NodeName.ANOVA_RM),
            normalize_node_id(NodeName.FRIEDMAN),
            normalize_node_id(NodeName.MCNEMAR),
            normalize_node_id(NodeName.CHOICE),
            normalize_node_id(NodeName.COX_REGRESSION),
            normalize_node_id(NodeName.DESCRIPTIVE_SUMMARY),
            normalize_node_id(NodeName.USER_INTERVENTION),
            normalize_node_id(NodeName.DESIGN_RECONCILIATION),
        ],
    ),
    WorkflowNode(
        id=normalize_node_id(NodeName.ASSESS_STUDY_DESIGN),
        label=NodeName.ASSESS_STUDY_DESIGN,
        kind='decision',
        transitions=[
            normalize_node_id(NodeName.NORMALITY_OF_DIFFERENCE),
            normalize_node_id(NodeName.TWO_INDEPENDENT_GROUPS),
            'end',
        ],
    ),
    WorkflowNode(
        id=normalize_node_id(NodeName.NORMALITY_OF_DIFFERENCE),
        label=NodeName.NORMALITY_OF_DIFFERENCE,
        kind='decision',
        transitions=[
            normalize_node_id(NodeName.PAIRED_T),
            normalize_node_id(NodeName.WILCOXON),
            'end',
        ],
    ),
    WorkflowNode(
        id=normalize_node_id(NodeName.TWO_INDEPENDENT_GROUPS),
        label=NodeName.TWO_INDEPENDENT_GROUPS,
        kind='decision',
        transitions=[
            normalize_node_id(NodeName.INDEP_T),
            normalize_node_id(NodeName.NONPARAMETRIC),
            normalize_node_id(NodeName.ANOVA_ASSUMPTIONS),
            'end',
        ],
    ),
    WorkflowNode(
        id=normalize_node_id(NodeName.ANOVA_ASSUMPTIONS),
        label=NodeName.ANOVA_ASSUMPTIONS,
        kind='decision',
        transitions=[
            normalize_node_id(NodeName.ANOVA_ONE_WAY),
            normalize_node_id(NodeName.KRUSKAL_WALLIS),
            'end',
        ],
    ),
    WorkflowNode(
        id=normalize_node_id(NodeName.PAIRED_T),
        label=NodeName.PAIRED_T,
        kind='test',
        transitions=[normalize_node_id(NodeName.SUMMARY)],
    ),
    WorkflowNode(
        id=normalize_node_id(NodeName.WILCOXON),
        label=NodeName.WILCOXON,
        kind='test',
        transitions=[normalize_node_id(NodeName.SUMMARY)],
    ),
    WorkflowNode(
        id=normalize_node_id(NodeName.INDEP_T),
        label=NodeName.INDEP_T,
        kind='test',
        transitions=[normalize_node_id(NodeName.SUMMARY)],
    ),
    WorkflowNode(
        id=normalize_node_id(NodeName.NONPARAMETRIC),
        label=NodeName.NONPARAMETRIC,
        kind='test',
        transitions=[normalize_node_id(NodeName.SUMMARY)],
    ),
    WorkflowNode(
        id=normalize_node_id(NodeName.WELCH),
        label=NodeName.WELCH,
        kind='test',
        transitions=[normalize_node_id(NodeName.SUMMARY)],
    ),
    WorkflowNode(
        id=normalize_node_id(NodeName.MANN),
        label=NodeName.MANN,
        kind='test',
        transitions=[normalize_node_id(NodeName.SUMMARY)],
    ),
    WorkflowNode(
        id=normalize_node_id(NodeName.ANOVA_ONE_WAY),
        label=NodeName.ANOVA_ONE_WAY,
        kind='test',
        transitions=[normalize_node_id(NodeName.SUMMARY)],
    ),
    WorkflowNode(
        id=normalize_node_id(NodeName.KRUSKAL_WALLIS),
        label=NodeName.KRUSKAL_WALLIS,
        kind='test',
        transitions=[normalize_node_id(NodeName.SUMMARY)],
    ),
    WorkflowNode(
        id=normalize_node_id(NodeName.ANOVA_RM),
        label=NodeName.ANOVA_RM,
        kind='test',
        transitions=[normalize_node_id(NodeName.SUMMARY)],
    ),
    WorkflowNode(
        id=normalize_node_id(NodeName.FRIEDMAN),
        label=NodeName.FRIEDMAN,
        kind='test',
        transitions=[normalize_node_id(NodeName.SUMMARY)],
    ),
    WorkflowNode(
        id=normalize_node_id(NodeName.MCNEMAR),
        label=NodeName.MCNEMAR,
        kind='test',
        transitions=[normalize_node_id(NodeName.SUMMARY)],
    ),
    WorkflowNode(
        id=normalize_node_id(NodeName.CHI2),
        label=NodeName.CHI2,
        kind='test',
        transitions=[normalize_node_id(NodeName.SUMMARY)],
    ),
    WorkflowNode(
        id=normalize_node_id(NodeName.FISHER),
        label=NodeName.FISHER,
        kind='test',
        transitions=[normalize_node_id(NodeName.SUMMARY)],
    ),
    WorkflowNode(
        id=normalize_node_id(NodeName.CHOICE),
        label=NodeName.CHOICE,
        kind='decision',
        transitions=[normalize_node_id(NodeName.ASSESS_STUDY_DESIGN)],
    ),
    WorkflowNode(
        id=normalize_node_id(NodeName.COX_REGRESSION),
        label=NodeName.COX_REGRESSION,
        kind='test',
        transitions=['end'],
    ),
    WorkflowNode(
        id=normalize_node_id(NodeName.DESCRIPTIVE_SUMMARY),
        label=NodeName.DESCRIPTIVE_SUMMARY,
        kind='report',
        transitions=['end'],
    ),
    WorkflowNode(
        id=normalize_node_id(NodeName.USER_INTERVENTION),
        label=NodeName.USER_INTERVENTION,
        kind='checkpoint',
        transitions=['end'],
    ),
    WorkflowNode(
        id=normalize_node_id(NodeName.SUMMARY),
        label=NodeName.SUMMARY,
        kind='aggregate',
        transitions=[normalize_node_id(NodeName.REVIEWER)],
    ),
    WorkflowNode(
        id=normalize_node_id(NodeName.REVIEWER),
        label=NodeName.REVIEWER,
        kind='review',
        transitions=['end'],
    ),
    WorkflowNode(id='end', label='End', kind='end', transitions=[]),
]

_EDGES: list[WorkflowEdge] = []
for node in _NODES:
    for target in node.transitions:
        _EDGES.append(WorkflowEdge(source=node.id, target=target))


def get_workflow_graph() -> dict[str, Any]:
    """Return a deep copy of the canonical workflow graph metadata."""
    return {
        'nodes': [node.__dict__ for node in _NODES],
        'edges': [edge.__dict__ for edge in _EDGES],
    }


def map_step_to_node_id(step_label: str | None) -> str | None:
    """Normalize a step label (or node name) to a workflow node id."""
    if not step_label:
        return None
    key = step_label.strip().lower()
    canonical = _NODE_ALIAS.get(key)
    if canonical:
        return normalize_node_id(canonical)
    # Try cleaning punctuation/underscores
    key = re.sub(r'[^a-z0-9]+', ' ', key).strip()
    if key in _NODE_ALIAS:
        return normalize_node_id(_NODE_ALIAS[key])
    return normalize_node_id(step_label)


def derive_progress(
    decision_steps: list[dict[str, Any]] | None,
    test_hierarchy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute visited and active nodes from recorded steps/test hierarchy."""
    known_nodes = {node.id for node in _NODES}
    visited: list[str] = []
    for step in decision_steps or []:
        node_id = step.get('node_id') or map_step_to_node_id(step.get('node')) or map_step_to_node_id(step.get('step'))
        if not node_id or node_id not in known_nodes:
            continue
        if not visited:
            visited.append('start')
        if node_id not in visited:
            visited.append(node_id)
    active_node = visited[-1] if visited else None

    selected_path = list(visited)
    chosen_test = None
    if test_hierarchy:
        chosen_test = test_hierarchy.get('chosen_test')
        if chosen_test:
            chosen_id = map_step_to_node_id(chosen_test)
            if chosen_id and chosen_id in known_nodes and chosen_id not in selected_path:
                selected_path.append(chosen_id)

    return {
        'visited_nodes': visited,
        'active_node': active_node,
        'selected_path': selected_path,
        'chosen_test': chosen_test,
    }


def workflow_payload(decision_steps: list[dict[str, Any]] | None, test_hierarchy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Combine metadata and state into a single payload."""
    graph = get_workflow_graph()
    progress = derive_progress(decision_steps, test_hierarchy)
    payload = deepcopy(graph)
    payload.update(progress)
    return payload
