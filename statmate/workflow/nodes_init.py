"""Initialization workflow nodes.

Covers: call_initialization_agent, design_verification_node,
design_reconciliation_node, assess_study_design_node.
"""

from typing import Any

import pandas as pd
from langchain_core.messages import AIMessage

from statmate.agents.auxiliary_agents import AssessDesignDeps, get_assess_design_study_agent
from statmate.agents.initial_insights_agent import (
    TOOL_FUNCS,
    build_partition_report,
    format_data_by_recommendation,
)
from statmate.core import NodeExecutionError, get_logger
from statmate.core.config import NodeName
from statmate.core.exceptions import RoutingError
from statmate.core.model_provider import execute_with_backoff
from statmate.core.validation import (
    StatisticalDesign,
    detect_wide_format_pairing,
    get_structural_summary,
    validate_statistical_design,
)
from statmate.workflow._node_helpers import _force_pairing_transform
from statmate.workflow.blueprint import build_data_blueprint
from statmate.workflow.initialization.column_role_agent import propose_column_roles
from statmate.workflow.initialization.route_proposal import propose_route
from statmate.workflow.initialization.structural_check import check_structural_validity
from statmate.workflow.model_factory import create_model, create_model_settings
from statmate.workflow.state import WorkflowState

logger = get_logger(__name__)


def call_initialization_agent(state: WorkflowState) -> WorkflowState:
    """Run the initialization agent to analyze data and suggest tests.

    Args:
        state: Current workflow state.

    Returns:
        Updated workflow state.

    Raises:
        NodeExecutionError: If the initialization fails.
    """
    try:
        structural = check_structural_validity(state.df)
        if not structural.is_valid:
            state.add_step(
                step="Structural Validation (pre-agent)",
                detail="Structural validation failed.",
                data={"errors": structural.errors},
            )
            raise NodeExecutionError(
                node_name="call_initialization_agent", original_error=ValueError("Structural validation failed")
            )

        state.statistical_design = structural.design
        state.paired = structural.design.is_paired if structural.design else state.paired
        state.comparison_matrix = structural.design.comparison_matrix if structural.design else state.comparison_matrix
        state.add_step(
            step="Structural Validation (pre-agent)",
            detail=structural.design.rationale if structural.design else "Structural design check completed.",
            data={
                "statistical_design": structural.design.as_dict() if structural.design else None,
                "structural_summary": structural.structural_summary,
            },
        )

        model = create_model(model_name=state.model_name, provider=state.provider)
        settings = create_model_settings(model_name=state.model_name)

        roles = propose_column_roles(
            df=state.df,
            structural=structural,
            model=model,
            model_settings=settings,
        )
        if roles.raw_response:
            state.add_step(
                step="Initialization Agent (raw)",
                detail="Captured raw column role agent output.",
                data={"response": roles.raw_response},
            )

        # Apply transformation if any
        if roles.data_transformation != "None":
            try:
                state.df = TOOL_FUNCS[roles.data_transformation](state.df, **roles.tool_arguments)
            except Exception as exc:
                logger.warning("Failed to apply transformation %s: %s", roles.data_transformation, exc)
                roles.data_transformation = "None"
                roles.tool_arguments = {}

        inp_df = state.df if isinstance(state.df, pd.DataFrame) else pd.DataFrame(state.df)

        validated_design = validate_statistical_design(
            inp_df,
            dependent_var=roles.analysis_columns or list(inp_df.columns),
            group_var=roles.group_column or (structural.design.grouping_variable if structural.design else None),
            subject_id=structural.design.subject_id_column if structural.design else None,
        )

        state.statistical_design = validated_design
        state.paired = validated_design.is_paired
        state.comparison_matrix = validated_design.comparison_matrix or state.comparison_matrix

        index_column = validated_design.subject_id_column or inp_df.index.name
        index_column = index_column if isinstance(index_column, str) else None
        target_column = (roles.analysis_columns or [validated_design.dependent_variable or None])[0]
        partition_report = build_partition_report(
            inp_df,
            roles.group_column or validated_design.grouping_variable,
            index_column,
        )

        blueprint = build_data_blueprint(
            inp_df,
            dependent_vars=roles.analysis_columns or list(inp_df.columns),
            group_var=roles.group_column or validated_design.grouping_variable,
            covariates=[],
            is_paired=validated_design.is_paired,
            index_column=index_column,
            target_column=target_column if isinstance(target_column, str) else None,
            partition_report=partition_report,
            raw_payload={
                "analysis_columns": roles.analysis_columns,
                "group_column": roles.group_column,
                "data_transformation": roles.data_transformation,
                "tool_arguments": roles.tool_arguments,
                "data_type": roles.data_type,
                "data_design": roles.data_design,
            },
        )
        state.attach_blueprint(blueprint)

        structural_text = (
            get_structural_summary(inp_df, roles.group_column) if roles.group_column else "No grouping column provided."
        )
        state.add_step(
            step="Structural Validation",
            detail=validated_design.rationale or "Structural design confirmed.",
            data={
                "statistical_design": validated_design.as_dict(),
                "structural_summary": structural.structural_summary,
                "structural_text": structural_text,
                "data_blueprint": blueprint.model_dump(),
            },
        )

        route = propose_route(structural=structural, roles=roles, blueprint=blueprint)
        state.pending_routing_decision = {
            "primary": route.primary,
            "alternatives": route.alternatives,
            "reason": route.metadata.get("reason"),
            "profile": route.metadata,
        }

        from statmate.agents.initial_insights_agent import InitialInsightsAgentResults
        from statmate.agents.initial_insights_agent import NodeName as AgentNodeName

        rec = InitialInsightsAgentResults(
            analysis_columns=roles.analysis_columns,
            group_column=roles.group_column,
            output_format="pd.DataFrame",
            data_analysis_result="Initialization pipeline completed.",
            route_to_test=[AgentNodeName(str(value)) for value in route.ordered_nodes],
            comments="Generated by phased initialization pipeline.",
            data_type=roles.data_type,
            data_design=roles.data_design,
            data_transformation=roles.data_transformation,
            tool_arguments=roles.tool_arguments,
            data_size=int(inp_df.shape[0]),
            number_of_columns=int(inp_df.shape[1]),
            variable_roles=[],
            distribution_metrics={},
            sample_balance=None,
            index_column=index_column if isinstance(index_column, str) else None,
            target_column=target_column if isinstance(target_column, str) else None,
            partition_report=partition_report,
        )

        formatted = format_data_by_recommendation(inp_df, rec, state.statistical_design)
        if isinstance(formatted, tuple):
            state.df, state.secondary_df = formatted
        else:
            state.df = formatted

        state.data_type = roles.data_type if roles.data_type in ("CONTINUOUS", "CATEGORICAL") else None
        state.target_columns = roles.analysis_columns if roles.analysis_columns else list(inp_df.columns)
        state.agent_design_hypothesis = roles.data_design
        state.add_result(AIMessage(content=str(rec)))
        state.add_step(
            step="Initialization",
            detail="Initialization pipeline completed.",
            data={
                "route_to_test": route.ordered_nodes,
                "data_type": roles.data_type,
                "analysis_columns": roles.analysis_columns,
                "group_column": roles.group_column,
                "data_transformation": roles.data_transformation,
                "tool_arguments": roles.tool_arguments,
                "data_design": roles.data_design,
            },
        )

        logger.info("Initialization pipeline completed.")
        return state
    except RoutingError as e:
        logger.error("Routing error in initialization: %s", e)
        raise
    except Exception as e:
        logger.error(f"Error in call_initialization_agent: {e}")
        raise NodeExecutionError(node_name="call_initialization_agent", original_error=e) from e


def design_verification_node(state: WorkflowState) -> WorkflowState:
    """Verify agent design hypothesis against the deterministic structural design.

    Contract: when a mismatch is detected, record it on ``state.design_verification`` and
    allow routing to proceed to ``DESIGN_RECONCILIATION`` rather than raising. This preserves
    the deterministic structural design for downstream routing while giving reconciliation a
    chance to resolve agent/structure disagreements.
    """
    try:
        design = state.statistical_design
        agent_design = state.agent_design_hypothesis
        mismatch = bool(design and agent_design and agent_design != design.design_type)

        detail = "Structural and agent designs are aligned."
        if mismatch and design:
            detail = (
                f"Design mismatch: validator={design.design_type} vs agent={agent_design}. Routing to reconciliation."
            )
        state.design_verification = {
            "mismatch": mismatch,
            "structural_design": design.as_dict() if design else None,
            "agent_design": agent_design,
        }
        if design:
            # Keep downstream routing consistent with deterministic structural design.
            state.paired = design.is_paired
        state.add_step(
            step="Design Verification",
            detail=detail,
            data=state.design_verification,
        )
        return state
    except Exception as e:
        logger.error(f"Error in design_verification_node: {e}")
        raise NodeExecutionError(node_name="design_verification", original_error=e) from e


def design_reconciliation_node(state: WorkflowState) -> WorkflowState:
    """Resolve structural mismatches, prioritizing wide-format pairing cues."""
    try:
        df = state.df if isinstance(state.df, pd.DataFrame) else pd.DataFrame(state.df)
        blueprint = state.data_blueprint
        verification = state.design_verification or {}
        wide_detection = detect_wide_format_pairing(list(df.columns))
        partition = blueprint.partition_report if blueprint else None
        overlap = (partition or {}).get("overlap", {})
        overlap_detected = bool(overlap.get("overlap_count"))

        # Prefer agent-specified grouping role when available
        group_col = None
        if blueprint and blueprint.variable_roles:
            for role in blueprint.variable_roles:
                if getattr(role, "role", "") == "group":
                    group_col = role.name
                    break
        if state.statistical_design and state.statistical_design.grouping_variable:
            group_col = group_col or state.statistical_design.grouping_variable

        resolved_design = state.statistical_design
        detail_parts: list[str] = []
        if wide_detection.get("detected"):
            detail_parts.append("Wide-format pairing detected from column names.")
            resolved_design = resolved_design or StatisticalDesign(
                design_type="paired",
                is_paired=True,
                grouping_variable=group_col,
                subject_id_column=getattr(state.statistical_design, "subject_id_column", None),
                rationale="",
                keyword_cues=state.statistical_design.keyword_cues if state.statistical_design else {},
            )
            resolved_design.design_type = "paired"
            resolved_design.is_paired = True
            resolved_design.rationale = (
                wide_detection.get("reason") or "Detected paired measurement columns; coercing to paired design."
            )
            resolved_design.overlap_summary = resolved_design.overlap_summary or {}
            resolved_design.overlap_summary["wide_format_pairs"] = wide_detection.get("pairs", [])
            state = _force_pairing_transform(state, df, wide_detection.get("pairs", []))
        elif overlap_detected and resolved_design and not resolved_design.is_paired:
            detail_parts.append("Subject overlap across groups indicates paired/mixed design.")
            resolved_design.design_type = "paired"
            resolved_design.is_paired = True
            resolved_design.overlap_summary = resolved_design.overlap_summary or {}
            resolved_design.overlap_summary["partition_overlap"] = overlap

        if resolved_design:
            state.statistical_design = resolved_design
            state.paired = resolved_design.is_paired
            if blueprint:
                updates: dict[str, Any] = {"is_paired": resolved_design.is_paired}
                if wide_detection.get("pairs") and state.secondary_df is not None:
                    samples = {
                        str(wide_detection["pairs"][0][0]): int(len(state.df)),
                        str(wide_detection["pairs"][0][1]): int(len(state.secondary_df)),
                    }
                    updates["group_samples"] = samples
                state.attach_blueprint(blueprint.model_copy(update=updates))

        state.design_verification = (verification or {}) | {"mismatch": False, "resolved": True}
        state.add_step(
            step=NodeName.DESIGN_RECONCILIATION,
            detail="; ".join(detail_parts) or "Design reconciliation completed.",
            data={
                "wide_format_detection": wide_detection,
                "partition_overlap": overlap,
                "resolved_design": resolved_design.as_dict() if resolved_design else None,
            },
        )
        return state
    except Exception as e:
        logger.error(f"Error in design_reconciliation_node: {e}")
        raise NodeExecutionError(node_name="design_reconciliation_node", original_error=e) from e


def assess_study_design_node(state: WorkflowState) -> WorkflowState:
    """Assess whether the study design is paired or independent.

    Args:
        state: Current workflow state.

    Returns:
        Updated workflow state.

    Raises:
        NodeExecutionError: If the assessment fails.
    """
    try:
        if state.statistical_design:
            state.paired = state.statistical_design.is_paired
            state.add_step(
                step="Assess Study Design",
                detail="Using structural validator output",
                data=state.statistical_design.as_dict(),
            )
            return state

        # Use model from state if specified
        model = create_model(model_name=state.model_name, provider=state.provider)
        settings = create_model_settings(model_name=state.model_name)
        agent = get_assess_design_study_agent(model=model, model_settings=settings)

        logger.info("assess_study_design_node\nmodel is fed with those data:")
        logger.info(state.results)

        res = execute_with_backoff(
            lambda: agent.run_sync(deps=AssessDesignDeps(msg=state.results)),
            on_retry=lambda attempt, delay, exc: state.add_step(
                step="Rate limit backoff",
                detail=f"Retrying study design check in {delay:.1f}s (attempt {attempt})",
                data={"error": str(exc)},
            ),
        )
        state.paired = res.output.paired

        msg = "~~~Paired comparison~~~" if res.output.paired else "~~~Two independent groups~~~"
        logger.info(msg)
        state.add_step(
            step="Assess Study Design",
            detail="Paired comparison" if res.output.paired else "Two independent groups",
            data={"paired": res.output.paired},
        )

        return state
    except Exception as e:
        logger.error(f"Error in assess_study_design_node: {e}")
        raise NodeExecutionError(node_name="assess_study_design_node", original_error=e) from e
