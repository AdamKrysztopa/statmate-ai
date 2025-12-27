"""Refactored statistical test workflow using modular components.

This is the new main entry point for the workflow, replacing the monolithic
statmate_flow.py with a cleaner API using the refactored modules.
"""

import os
import sys
from typing import Any, Callable

import pandas as pd

from statmate.core.config import Config, default_config
from statmate.core.pii import sanitize_dataframe
from statmate.core.logging_config import get_logger, setup_logging
from statmate.workflow.graph_builder import build_workflow_graph, create_default_checkpointer
from statmate.workflow.model_factory import initialize_default_factory
from statmate.workflow.state import WorkflowState, create_initial_state

logger = get_logger(__name__)


class StatMateWorkflow:
    """Main workflow class for StatMate statistical analysis."""

    def __init__(self, config: Config | None = None, checkpointer: Any | None = None):
        """Initialize the workflow.

        Args:
            config: Configuration object. If None, uses default_config.
            checkpointer: Optional LangGraph checkpointer to persist state.
        """
        self.config = config or default_config
        setup_logging(self.config.logging)
        self.checkpointer = checkpointer or create_default_checkpointer()

        # Initialize the global model factory with this workflow's config
        if hasattr(self.config, 'model'):
            initialize_default_factory(self.config.model)
        else:
            logger.warning("Config has no 'model' attribute; model factory will use defaults.")

        self.graph = build_workflow_graph(checkpointer=self.checkpointer)
        logger.info('StatMate workflow initialized')

    def run(
        self,
        data: pd.DataFrame | pd.Series,
        target_columns: list[str] | None = None,
        paired: bool = False,
        do_association: bool = False,
        model_name: str | None = None,
        provider: str | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
        thread_id: str | None = None,
    ) -> WorkflowState:
        """Run the statistical test workflow on the provided data.

        Args:
            data: Input data for analysis.
            target_columns: Columns to analyze. If None, uses all columns.
            paired: Whether data represents paired measurements.
            do_association: Whether to perform association tests.
            model_name: Override for the AI model to use.
            provider: Override for the model provider.
            thread_id: Optional identifier used by LangGraph checkpointers to resume runs.

        Returns:
            Final workflow state with results.
        """
        logger.info('Starting workflow execution')
        logger.info('Data shape: %s', data.shape if hasattr(data, 'shape') else len(data))

        # Determine model and provider to use
        final_model_name = model_name
        final_provider = provider

        # Use config defaults if no overrides are provided
        if hasattr(self.config, 'model'):
            if final_model_name is None:
                final_model_name = self.config.model.default_model_name

        # Sanitize data before any LLM calls
        original_series = isinstance(data, pd.Series)
        base_frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        sanitized_df, mask_report = sanitize_dataframe(base_frame)
        if original_series and isinstance(sanitized_df, pd.DataFrame):
            sanitized_df = sanitized_df.iloc[:, 0]

        # Create initial state
        initial_state = create_initial_state(
            df=sanitized_df,
            target_columns=target_columns,
            paired=paired,
            do_association=do_association,
            model_name=final_model_name,
            provider=final_provider,
        )
        if mask_report.get('masked_columns'):
            initial_state.add_assumption_entry({'node': 'PII masking', **mask_report})
            initial_state.add_step(
                step='Input Sanitization',
                detail='Applied PII masking before agent calls.',
                data=mask_report,
            )

        # Run workflow
        try:
            stream_config: dict[str, Any] = {}
            if thread_id:
                stream_config['configurable'] = {'thread_id': thread_id}

            def _run_graph(state: WorkflowState) -> WorkflowState:
                """Execute the compiled graph and return the final state."""
                final_state_result = None
                for state_update in self.graph.stream(state, config=stream_config or None):
                    final_state_result = state_update
                    if on_update:
                        try:
                            on_update(state_update)
                        except Exception as callback_error:
                            logger.warning('Streaming callback failed: %s', callback_error)

                if final_state_result is None or not isinstance(final_state_result, dict):
                    logger.warning('Workflow did not produce a final state dictionary. Returning initial state.')
                    return state

                return list(final_state_result.values())[0]

            try:
                final_state = _run_graph(initial_state)
            except Exception as exc:
                # LangGraph's SqliteSaver uses msgpack, which cannot serialize WorkflowState/DataFrames.
                # If checkpoint serialization fails, fall back to an in-memory graph without a checkpointer.
                message = str(exc).lower()
                if 'msgpack' in message and 'serializable' in message:
                    logger.warning('Checkpoint serialization failed (%s); retrying without a checkpointer', exc)
                    self.checkpointer = None
                    self.graph = build_workflow_graph(checkpointer=None)
                    fresh_state = initial_state.model_copy(deep=True)
                    final_state = _run_graph(fresh_state)
                else:
                    raise

            logger.info('Workflow execution completed successfully')
            return final_state

        except Exception as e:
            logger.error('Workflow execution failed: %s', e)
            raise

    def visualize(self, output_path: str = 'workflow_graph.md') -> str:
        """Generate a visualization of the workflow graph.

        Args:
            output_path: Path to save the Mermaid diagram.

        Returns:
            Mermaid diagram as a string.
        """
        try:
            mermaid_src = self.graph.get_graph().draw_mermaid()
            md = f'```mermaid\n{mermaid_src}\n```'

            with open(output_path, 'w') as f:
                f.write(md)

            logger.info('Saved workflow diagram to %s', output_path)
            return md
        except Exception as e:
            logger.error('Failed to generate workflow visualization: %s', e)
            raise


def run_workflow(
    data: pd.DataFrame | pd.Series,
    target_columns: list[str] | None = None,
    paired: bool = False,
    do_association: bool = False,
    model_name: str | None = None,
    provider: str | None = None,
    config: Config | None = None,
) -> WorkflowState:
    """Convenience function to run the workflow.

    Args:
        data: Input data for analysis.
        target_columns: Columns to analyze. If None, uses all columns.
        paired: Whether data represents paired measurements.
        do_association: Whether to perform association tests.
        model_name: Override for the AI model to use.
        provider: Override for the model provider.
        config: Configuration object. If None, uses default_config.

    Returns:
        Final workflow state with results.
    """
    workflow = StatMateWorkflow(config=config)
    return workflow.run(
        data=data,
        target_columns=target_columns,
        paired=paired,
        do_association=do_association,
        model_name=model_name,
        provider=provider,
    )


if __name__ == '__main__':
    import numpy as np

    # Check for API key before running the example
    if not os.getenv('OPENAI_API_KEY'):
        print('ERROR: The OPENAI_API_KEY environment variable is not set.')
        print('Please set it to your OpenAI API key to run this example.')
        print('Example: export OPENAI_API_KEY="your-key-here"')
        sys.exit(1)

    # Example usage
    logger.info('=== StatMate Workflow Example ===')

    # Example 1: Two independent groups with different variances
    n = 250
    male_performance = np.random.normal(loc=50, scale=5, size=n)
    female_performance = np.random.normal(loc=50, scale=20, size=n)

    df = pd.DataFrame(
        {'gender': ['Male'] * n + ['Female'] * n, 'performance': np.concatenate([male_performance, female_performance])}
    )

    workflow = StatMateWorkflow()

    # Visualize the workflow
    workflow.visualize()

    # Run the workflow
    result = workflow.run(df)

    # Print results
    print('\n=== Workflow Results ===')
    if result:
        print(f'Number of tests performed: {len(result.get("probabilities"))}')
        print(f'Tests: {list(result.get("probabilities").keys())}')
        print('\n=== Results Messages ===')
        for i, msg in enumerate(result.get('results', {}), 1):
            print(f'\n--- Message {i} ---')
            print(msg.content)
    else:
        print('Workflow did not return a final state.')
