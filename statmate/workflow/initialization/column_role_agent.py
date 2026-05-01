"""Phase 2: column role proposal agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import logging

import pandas as pd
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import Model
from pydantic_ai.settings import ModelSettings

logger = logging.getLogger(__name__)

from statmate.agents.initial_insights_agent import (
    InitialInsightsAgentDeps,
    NodeName as AgentNodeName,
    TOOL_FUNCS,
    validate_tool_args,
)
from statmate.workflow.initialization.structural_check import StructuralCheckResult

COLUMN_ROLE_PROMPT = """
You are a column role assignment assistant.

Given:
- Structural summary (types, cardinality)
- Design hint (paired/independent/mixed)

Tasks:
1) Identify analysis_columns (numeric variables for testing).
2) Identify group_column (categorical grouping variable).
3) Determine if transformation needed (wide→long, contingency table).
4) Specify tool_arguments for transformation.
5) Propose data_type (CONTINUOUS/CATEGORICAL) and data_design (independent/paired/mixed).

Rules:
- group_column must not be part of analysis_columns.
- For paired design, look for repeated measures structure.
- For categorical analysis, identify row/column categories.
- Always include tool_arguments ({} if no transform).

Output JSON with:
analysis_columns, group_column, data_transformation, tool_arguments, data_type, data_design, route_to_test.
route_to_test must use NodeName enum values.
"""


@dataclass
class ColumnRoleResult:
    analysis_columns: list[str]
    group_column: str | None
    data_transformation: str
    tool_arguments: dict[str, Any]
    data_type: str
    data_design: str
    route_to_test: list[AgentNodeName]
    raw_response: dict[str, Any] | None = None


def build_column_role_agent(
    model: Model,
    *,
    system_prompt: str = COLUMN_ROLE_PROMPT,
    model_settings: ModelSettings | None = None,
) -> Agent[InitialInsightsAgentDeps, dict[str, Any]]:
    """Create a constrained agent for column role assignment."""
    return Agent(
        model=model,
        model_settings=model_settings,
        deps_type=InitialInsightsAgentDeps,
        result_type=dict[str, Any],
        name='Column Role Agent',
        system_prompt=system_prompt,
    )


def propose_column_roles(
    *,
    df: pd.DataFrame | pd.Series,
    structural: StructuralCheckResult,
    model: Model,
    model_settings: ModelSettings | None,
) -> ColumnRoleResult:
    """Run the column role agent and return structured output."""
    frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    agent = build_column_role_agent(model, model_settings=model_settings)

    structural_payload = {
        'statistical_design': structural.design.as_dict() if structural.design else None,
        'structural_summary': structural.structural_summary,
    }
    prompt = (
        'Assign column roles based on the structural summary. '
        f'STRUCTURAL SUMMARY: {structural_payload}'
    )

    try:
        response = agent.run_sync(
            user_prompt=prompt,
            deps=InitialInsightsAgentDeps(
                user_input='Assign column roles.',
                input_data=frame,
                columns_decision=list(frame.columns),
            ),
        )
        data = response.data or {}
    except Exception as exc:
        logger.warning('Column role agent failed, falling back to defaults: %s', exc)
        data = {}

    analysis_columns = list(data.get('analysis_columns') or [])
    group_column = data.get('group_column')
    data_transformation = data.get('data_transformation') or 'None'
    tool_arguments = data.get('tool_arguments') or {}
    data_type = data.get('data_type') or 'CONTINUOUS'
    data_design = data.get('data_design') or (structural.design.design_type if structural.design else 'independent')
    route_to_test = data.get('route_to_test') or []

    if not analysis_columns:
        analysis_columns = [str(col) for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col])]
    if not analysis_columns:
        analysis_columns = [str(col) for col in frame.columns]

    if group_column and group_column in analysis_columns:
        analysis_columns = [col for col in analysis_columns if col != group_column]

    if not group_column and structural.design and structural.design.grouping_variable:
        group_column = structural.design.grouping_variable

    if data_transformation not in ('None', 'transform_independent', 'transform_categorical'):
        data_transformation = 'None'
        tool_arguments = {}

    if data_type not in ('CATEGORICAL', 'CONTINUOUS'):
        data_type = 'CONTINUOUS'

    if data_design not in ('independent', 'paired', 'mixed'):
        data_design = structural.design.design_type if structural.design else 'independent'

    if data_transformation != 'None':
        try:
            tool_arguments = validate_tool_args(data_transformation, tool_arguments)
            frame = TOOL_FUNCS[data_transformation](frame, **tool_arguments)
        except Exception as exc:
            logger.warning('Skipping transformation %s due to invalid tool args: %s', data_transformation, exc)
            data_transformation = 'None'
            tool_arguments = {}

    safe_route: list[AgentNodeName] = []
    for value in route_to_test:
        try:
            safe_route.append(AgentNodeName(value) if isinstance(value, str) else value)
        except Exception:
            logger.warning('Dropping invalid route_to_test entry: %s', value)

    return ColumnRoleResult(
        analysis_columns=analysis_columns,
        group_column=group_column,
        data_transformation=data_transformation,
        tool_arguments=tool_arguments,
        data_type=data_type,
        data_design=data_design,
        route_to_test=safe_route,
        raw_response=data,
    )
