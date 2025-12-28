"""Routing and decision engine utilities for workflow edges."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from langgraph.graph import END

from statmate.core import get_logger
from statmate.core.config import DataType, NodeName, default_config
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


class DecisionEngine:
    """Deterministic, registry-driven routing engine."""

    def __init__(self, registry: MethodRegistry | None = None):
        self.registry = registry or _build_default_registry()

    def _blueprint(self, state: WorkflowState) -> DataBlueprint | None:
        return getattr(state, 'data_blueprint', None)

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

    def _group_count(self, state: WorkflowState, blueprint: DataBlueprint | None) -> int | None:
        if blueprint and blueprint.sample_balance and blueprint.sample_balance.group_sizes:
            return len(blueprint.sample_balance.group_sizes)
        if state.secondary_df is not None:
            return 2
        return None

    def _scale(self, state: WorkflowState, blueprint: DataBlueprint | None) -> Literal['continuous', 'categorical', 'survival']:
        if blueprint and blueprint.survival_data:
            return 'survival'
        if state.data_type == DataType.SURVIVAL:
            return 'survival'
        if state.data_type == DataType.CATEGORICAL:
            return 'categorical'
        return 'continuous'

    def _effective_n(self, state: WorkflowState, blueprint: DataBlueprint | None) -> int:
        if blueprint and blueprint.sample_balance and blueprint.sample_balance.group_sizes:
            return sum(blueprint.sample_balance.group_sizes.values())
        return int(state.number_of_samples or 0)

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
        normal_flag = self._normal_flag(state, blueprint)
        group_count = self._group_count(state, blueprint)
        paired = state.paired if state.paired is not None else (state.statistical_design.is_paired if state.statistical_design else None)
        scale = self._scale(state, blueprint)

        # Hard constraints
        if scale == 'survival':
            return NodeName.COX_REGRESSION
        if scale == 'categorical' and paired:
            return NodeName.MCNEMAR
        if scale == 'continuous' and normal_flag is False and group_count == 2:
            return NodeName.NONPARAMETRIC

        if scale == 'categorical':
            eff_n = self._effective_n(state, blueprint)
            return NodeName.CHI2 if eff_n > categorical_threshold else NodeName.FISHER

        profile = MethodProfile(scale, normal_flag, group_count, paired)
        suggestion = self.registry.suggest(profile)
        if not suggestion:
            logger.warning('No registry suggestion for profile %s; defaulting to study design assessment.', profile)
            return NodeName.ASSESS_STUDY_DESIGN

        if assumption_status and assumption_status.get('status') == 'fail' and suggestion.alternatives:
            state.pending_routing_decision = {
                'profile': asdict(profile),
                'primary': suggestion.primary,
                'alternatives': suggestion.alternatives,
                'reason': suggestion.reason,
                'assumption_status': assumption_status,
                'selected': suggestion.alternatives[0],
            }
            return suggestion.alternatives[0]

        pending = {
            'profile': asdict(profile),
            'primary': suggestion.primary,
            'alternatives': suggestion.alternatives,
            'reason': suggestion.reason,
            'assumption_status': assumption_status,
        }
        state.pending_routing_decision = pending

        if scale == 'continuous' and not prefer_terminal:
            # Preserve the existing assumption-check path while keeping registry hints attached
            return NodeName.ASSESS_STUDY_DESIGN

        if suggestion.alternatives:
            return NodeName.CHOICE

        if assumption_status and assumption_status.get('status') == 'fail' and suggestion.alternatives:
            return suggestion.alternatives[0]

        return suggestion.primary


default_registry = _build_default_registry()
decision_engine = DecisionEngine(default_registry)


def decide_outcome(state: WorkflowState, categorical_threshold: int | None = None) -> str:
    """Determine the initial routing using the registry-driven engine."""
    try:
        return decision_engine.evaluate_routing(state, categorical_threshold=categorical_threshold)
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
