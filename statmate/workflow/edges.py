"""Routing and decision engine utilities for workflow edges."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from langgraph.graph import END

from statmate.core import get_logger
from statmate.core.config import DataType, NodeName, default_config
from statmate.core.exceptions import RoutingError
from statmate.workflow.blueprint import DataBlueprint
from statmate.workflow.state import WorkflowState

logger = get_logger(__name__)


@dataclass(frozen=True)
class MethodProfile:
    """Describe a statistical testing context."""

    scale: Literal['continuous', 'categorical', 'survival']
    normal: bool | None
    groups: int | None
    paired: bool | None


@dataclass
class MethodSuggestion:
    """Suggested primary method and backups."""

    primary: str
    alternatives: list[str]
    weight: float = 1.0
    reason: str | None = None


@dataclass(frozen=True)
class NodeMetadata:
    """Hard constraints for routing candidates."""

    is_paired: bool | None = None
    min_sample_size: int = 0
    group_count_range: tuple[int | None, int | None] | None = None


class MethodRegistry:
    """Registry mapping profiles to weighted suggestions."""

    def __init__(self) -> None:
        self._registry: dict[MethodProfile, list[MethodSuggestion]] = {}

    def register(self, profile: MethodProfile, suggestion: MethodSuggestion) -> None:
        bucket = self._registry.setdefault(profile, [])
        bucket.append(suggestion)
        bucket.sort(key=lambda s: s.weight, reverse=True)

    def suggest(self, profile: MethodProfile) -> MethodSuggestion | None:
        return self._registry.get(profile, [None])[0]


def _build_default_registry() -> MethodRegistry:
    registry = MethodRegistry()
    registry.register(
        MethodProfile('continuous', True, 2, False),
        MethodSuggestion(NodeName.INDEP_T, [NodeName.WELCH, NodeName.MANN], weight=1.0, reason='Normal & equal var'),
    )
    registry.register(
        MethodProfile('continuous', False, 2, False),
        MethodSuggestion(NodeName.NONPARAMETRIC, [NodeName.WELCH, NodeName.MANN], weight=1.0, reason='Non-normal'),
    )
    registry.register(
        MethodProfile('continuous', True, 2, True),
        MethodSuggestion(NodeName.PAIRED_T, [NodeName.WILCOXON], weight=1.0, reason='Paired & normal'),
    )
    registry.register(
        MethodProfile('continuous', False, 2, True),
        MethodSuggestion(NodeName.WILCOXON, [NodeName.PAIRED_T], weight=1.0, reason='Paired & non-normal'),
    )
    registry.register(
        MethodProfile('categorical', None, None, False),
        MethodSuggestion(NodeName.CHI2, [NodeName.FISHER], weight=1.0, reason='Categorical'),
    )
    registry.register(
        MethodProfile('categorical', None, None, True),
        MethodSuggestion(NodeName.MCNEMAR, [NodeName.CHI2], weight=1.0, reason='Paired categorical'),
    )
    registry.register(
        MethodProfile('survival', None, None, False),
        MethodSuggestion(NodeName.COX_REGRESSION, [], weight=1.0, reason='Survival outcome'),
    )
    return registry


NODE_METADATA: dict[str, NodeMetadata] = {
    NodeName.PAIRED_T: NodeMetadata(is_paired=True, min_sample_size=2, group_count_range=(1, 2)),
    NodeName.WILCOXON: NodeMetadata(is_paired=True, min_sample_size=2, group_count_range=(1, 2)),
    NodeName.INDEP_T: NodeMetadata(is_paired=False, min_sample_size=2, group_count_range=(2, 2)),
    NodeName.WELCH: NodeMetadata(is_paired=False, min_sample_size=2, group_count_range=(2, 2)),
    NodeName.MANN: NodeMetadata(is_paired=False, min_sample_size=2, group_count_range=(2, 2)),
    NodeName.NONPARAMETRIC: NodeMetadata(is_paired=False, min_sample_size=2, group_count_range=(2, 2)),
    NodeName.ANOVA_ASSUMPTIONS: NodeMetadata(is_paired=False, min_sample_size=2, group_count_range=(3, None)),
    NodeName.ANOVA_ONE_WAY: NodeMetadata(is_paired=False, min_sample_size=2, group_count_range=(3, None)),
    NodeName.KRUSKAL_WALLIS: NodeMetadata(is_paired=False, min_sample_size=2, group_count_range=(3, None)),
    NodeName.ANOVA_RM: NodeMetadata(is_paired=True, min_sample_size=2, group_count_range=(3, None)),
    NodeName.FRIEDMAN: NodeMetadata(is_paired=True, min_sample_size=2, group_count_range=(3, None)),
    NodeName.CHI2: NodeMetadata(is_paired=False, min_sample_size=2, group_count_range=(2, None)),
    NodeName.FISHER: NodeMetadata(is_paired=False, min_sample_size=2, group_count_range=(2, None)),
    NodeName.MCNEMAR: NodeMetadata(is_paired=True, min_sample_size=2, group_count_range=(2, None)),
    NodeName.COX_REGRESSION: NodeMetadata(is_paired=None, min_sample_size=1, group_count_range=None),
    NodeName.DESCRIPTIVE_SUMMARY: NodeMetadata(is_paired=None, min_sample_size=0, group_count_range=None),
    NodeName.USER_INTERVENTION: NodeMetadata(is_paired=None, min_sample_size=0, group_count_range=None),
}


class SufficiencyValidator:
    """Pre-flight guardrail for minimum per-group sample sizes."""

    def __init__(self, min_group_size: int = 2):
        self.min_group_size = min_group_size

    def check(self, blueprint: DataBlueprint | None) -> tuple[bool, dict[str, int]]:
        if not blueprint:
            return True, {}
        counts: dict[str, int] = {}
        if blueprint.group_samples:
            counts = blueprint.group_samples
        elif blueprint.sample_balance and blueprint.sample_balance.group_sizes:
            counts = blueprint.sample_balance.group_sizes

        if not counts:
            return True, {}

        insufficient = {group: n for group, n in counts.items() if n < self.min_group_size}
        return len(insufficient) == 0, insufficient


class DecisionEngine:
    """Deterministic, registry-driven routing engine."""

    def __init__(self, registry: MethodRegistry | None = None):
        self.registry = registry or _build_default_registry()
        self.metadata = NODE_METADATA
        self.sufficiency_validator = SufficiencyValidator()

    def _blueprint(self, state: WorkflowState) -> DataBlueprint | None:
        return getattr(state, 'data_blueprint', None)

    def _paired_flag(self, state: WorkflowState, blueprint: DataBlueprint | None) -> bool | None:
        if blueprint and blueprint.is_paired is not None:
            return blueprint.is_paired
        if state.paired is not None:
            return state.paired
        if state.statistical_design:
            return state.statistical_design.is_paired
        return None

    def _normal_flag(self, state: WorkflowState, blueprint: DataBlueprint | None) -> bool | None:
        if blueprint:
            for metric in blueprint.distribution_metrics.values():
                p_val = metric.normality_p_value
                if p_val is not None:
                    return p_val >= default_config.statistical.normality_threshold

        # Fall back to logged p-values if blueprint lacks info
        p_normality = state.get_probability('normality_of_difference', None)
        if p_normality is not None:
            return p_normality >= default_config.statistical.normality_threshold
        return None

    def _group_samples(self, blueprint: DataBlueprint | None) -> dict[str, int]:
        if blueprint and blueprint.group_samples:
            return blueprint.group_samples
        if blueprint and blueprint.sample_balance and blueprint.sample_balance.group_sizes:
            return {str(k): int(v) for k, v in blueprint.sample_balance.group_sizes.items()}
        return {}

    def _group_count(self, state: WorkflowState, blueprint: DataBlueprint | None) -> int | None:
        if blueprint:
            if blueprint.group_samples and set(blueprint.group_samples) != {'__all__'}:
                return len(blueprint.group_samples)
            if blueprint.sample_balance and blueprint.sample_balance.group_sizes:
                return len(blueprint.sample_balance.group_sizes)
        if state.secondary_df is not None:
            return 2
        return None

    def _min_group_size(self, blueprint: DataBlueprint | None) -> int | None:
        counts = self._group_samples(blueprint)
        return min(counts.values()) if counts else None

    def _scale(self, state: WorkflowState, blueprint: DataBlueprint | None) -> Literal['continuous', 'categorical', 'survival']:
        if blueprint and blueprint.survival_data:
            return 'survival'
        if state.data_type == DataType.SURVIVAL:
            return 'survival'
        if state.data_type == DataType.CATEGORICAL:
            return 'categorical'
        return 'continuous'

    def _effective_n(self, state: WorkflowState, blueprint: DataBlueprint | None) -> int:
        if blueprint:
            if blueprint.group_samples:
                return sum(blueprint.group_samples.values())
            if blueprint.sample_balance and blueprint.sample_balance.group_sizes:
                return sum(blueprint.sample_balance.group_sizes.values())
        return int(state.number_of_samples or 0)

    def _get_metadata(self, node: str) -> NodeMetadata:
        return self.metadata.get(node, NodeMetadata())

    def _node_allowed(
        self,
        node: str,
        *,
        blueprint: DataBlueprint | None,
        group_count: int | None,
        min_group_size: int | None,
        paired_flag: bool | None,
    ) -> bool:
        meta = self._get_metadata(node)
        if paired_flag is True and meta.is_paired is False:
            return False
        if paired_flag is False and meta.is_paired is True:
            return False
        if meta.group_count_range and group_count is not None:
            low, high = meta.group_count_range
            if low is not None and group_count < low:
                return False
            if high is not None and group_count > high:
                return False
        if meta.min_sample_size and min_group_size is not None and min_group_size < meta.min_sample_size:
            return False
        return True

    def _filter_candidates(
        self,
        candidates: list[str],
        *,
        blueprint: DataBlueprint | None,
        group_count: int | None,
        min_group_size: int | None,
        paired_flag: bool | None,
    ) -> list[str]:
        seen: set[str] = set()
        allowed: list[str] = []
        for node in candidates:
            if node in seen:
                continue
            seen.add(node)
            if self._node_allowed(
                node, blueprint=blueprint, group_count=group_count, min_group_size=min_group_size, paired_flag=paired_flag
            ):
                allowed.append(node)
        return allowed

    def evaluate_routing(
        self,
        state: WorkflowState,
        assumption_status: dict[str, str] | None = None,
        categorical_threshold: int | None = None,
        prefer_terminal: bool = False,
    ) -> str:
        """Evaluate the next node based on blueprint + registry."""
        if categorical_threshold is None:
            categorical_threshold = default_config.statistical.categorical_sample_size_threshold

        blueprint = self._blueprint(state)
        sufficient, deficits = self.sufficiency_validator.check(blueprint)
        if not sufficient:
            state.pending_routing_decision = {
                'reason': 'insufficient_samples',
                'deficits': deficits,
                'profile': None,
            }
            return NodeName.DESCRIPTIVE_SUMMARY

        verification = getattr(state, 'design_verification', None)
        if verification and verification.get('mismatch'):
            state.pending_routing_decision = {
                'reason': 'design_mismatch',
                'structural_design': verification.get('structural_design'),
                'agent_design': verification.get('agent_design'),
            }
            return NodeName.DESIGN_RECONCILIATION

        normal_flag = self._normal_flag(state, blueprint)
        group_count = self._group_count(state, blueprint)
        paired = self._paired_flag(state, blueprint)
        scale = self._scale(state, blueprint)
        min_group_size = self._min_group_size(blueprint)

        # Hard constraints
        if scale == 'survival':
            return NodeName.COX_REGRESSION
        if scale == 'categorical' and paired:
            return NodeName.MCNEMAR
        if scale == 'continuous' and group_count and group_count > 2:
            if paired:
                return NodeName.ANOVA_RM if normal_flag is not False else NodeName.FRIEDMAN
            if assumption_status and assumption_status.get('status') == 'fail':
                return NodeName.KRUSKAL_WALLIS
            if normal_flag is False and prefer_terminal:
                return NodeName.KRUSKAL_WALLIS
            return NodeName.ANOVA_ASSUMPTIONS if not prefer_terminal else NodeName.ANOVA_ONE_WAY
        if scale == 'continuous' and normal_flag is False and group_count == 2 and not paired:
            return NodeName.NONPARAMETRIC

        if scale == 'categorical':
            eff_n = self._effective_n(state, blueprint)
            return NodeName.CHI2 if eff_n > categorical_threshold else NodeName.FISHER

        profile = MethodProfile(scale, normal_flag, group_count, paired)
        suggestion = self.registry.suggest(profile)
        if not suggestion:
            logger.warning('No registry suggestion for profile %s; defaulting to study design assessment.', profile)
            return NodeName.ASSESS_STUDY_DESIGN

        primary_meta = self._get_metadata(suggestion.primary)
        if paired and primary_meta.is_paired is False:
            state.pending_routing_decision = {
                'profile': asdict(profile),
                'primary': suggestion.primary,
                'alternatives': suggestion.alternatives,
                'reason': 'paired_data_requires_paired_node',
            }
            return NodeName.DESIGN_RECONCILIATION

        candidates: list[str] = []
        if assumption_status and assumption_status.get('status') == 'fail' and suggestion.alternatives:
            candidates.append(suggestion.alternatives[0])
        candidates.append(suggestion.primary)
        candidates.extend(suggestion.alternatives)

        filtered = self._filter_candidates(
            candidates,
            blueprint=blueprint,
            group_count=group_count,
            min_group_size=min_group_size,
            paired_flag=paired,
        )

        pending = {
            'profile': asdict(profile),
            'primary': suggestion.primary,
            'alternatives': suggestion.alternatives,
            'reason': suggestion.reason,
            'assumption_status': assumption_status,
            'filtered_candidates': filtered,
        }
        state.pending_routing_decision = pending

        if not filtered:
            pending['reason'] = pending.get('reason') or 'no_candidate_after_constraints'
            pending['deficits'] = deficits
            return NodeName.USER_INTERVENTION

        selected = filtered[0]
        state.pending_routing_decision = {**pending, 'selected': selected}

        if scale == 'continuous' and not prefer_terminal:
            # Preserve the existing assumption-check path while keeping registry hints attached
            return NodeName.ASSESS_STUDY_DESIGN

        if assumption_status and assumption_status.get('status') == 'fail':
            return selected

        if len(filtered) > 1 and suggestion.alternatives:
            return NodeName.CHOICE

        return selected


default_registry = _build_default_registry()
decision_engine = DecisionEngine(default_registry)


def decide_outcome(state: WorkflowState, categorical_threshold: int | None = None) -> str:
    """Determine the initial routing using the registry-driven engine."""
    try:
        return decision_engine.evaluate_routing(state, categorical_threshold=categorical_threshold)
    except RoutingError as exc:
        logger.error('Routing error: %s', exc)
        state.pending_routing_decision = (state.pending_routing_decision or {}) | {'error': str(exc)}
        return NodeName.USER_INTERVENTION
    except Exception as e:  # pragma: no cover - defensive
        logger.error(f'Error in decide_outcome: {e}')
        return END


def assess_study_design(state: WorkflowState) -> str:
    """Branch on paired vs independent groups."""
    try:
        return NodeName.NORMALITY_OF_DIFFERENCE if state.paired else NodeName.TWO_INDEPENDENT_GROUPS
    except Exception as e:
        logger.error(f'Error in assess_study_design: {e}')
        return END


def parametric_assumptions(state: WorkflowState, alpha: float | None = None) -> str:
    """Decide between parametric and non-parametric tests for paired data."""
    if alpha is None:
        alpha = default_config.statistical.normality_threshold

    try:
        p_normality = state.get_probability('normality_of_difference', 0)
        return NodeName.PAIRED_T if p_normality > alpha else NodeName.WILCOXON
    except Exception as e:
        logger.error(f'Error in parametric_assumptions: {e}')
        return END


def decide_two_independent(state: WorkflowState, alpha: float | None = None) -> str:
    """Choose between parametric and non-parametric tests for independent groups."""
    if alpha is None:
        alpha = default_config.statistical.variance_threshold

    try:
        p_shapiro1 = state.get_probability('shapiro_group1', 0)
        p_shapiro2 = state.get_probability('shapiro_group2', 0)
        p_levene = state.get_probability('levene', 0)

        if p_shapiro1 > alpha and p_shapiro2 > alpha and p_levene > alpha:
            return NodeName.INDEP_T
        return NodeName.NONPARAMETRIC
    except Exception as e:
        logger.error(f'Error in decide_two_independent: {e}')
        return END


def decide_anova_path(state: WorkflowState, alpha: float | None = None) -> str:
    """Route to One-way ANOVA or Kruskal-Wallis based on assumption checks."""
    if alpha is None:
        alpha = default_config.statistical.variance_threshold

    try:
        p_levene = state.get_probability('anova_levene', 0)
        p_shapiro = state.get_probability('anova_min_shapiro', default_config.statistical.normality_threshold)

        if p_levene > alpha and p_shapiro > default_config.statistical.normality_threshold:
            return NodeName.ANOVA_ONE_WAY
        return NodeName.KRUSKAL_WALLIS
    except Exception as e:
        logger.error(f'Error in decide_anova_path: {e}')
        return END
