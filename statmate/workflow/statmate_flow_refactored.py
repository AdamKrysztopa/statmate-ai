"""Refactored statistical test workflow using modular components.

This is the new main entry point for the workflow, replacing the monolithic
statmate_flow.py with a cleaner API using the refactored modules.
"""

import pandas as pd

from statmate.config import Config, default_config
from statmate.logging_config import get_logger, setup_logging
from statmate.workflow.graph_builder import build_workflow_graph
from statmate.workflow.state import WorkflowState, create_initial_state

logger = get_logger(__name__)


class StatMateWorkflow:
    """Main workflow class for StatMate statistical analysis."""

    def __init__(self, config: Config | None = None):
        """Initialize the workflow.

        Args:
            config: Configuration object. If None, uses default_config.
        """
        self.config = config or default_config
        setup_logging(self.config.logging)
        self.graph = build_workflow_graph()
        logger.info('StatMate workflow initialized')

    def run(
        self,
        data: pd.DataFrame | pd.Series,
        target_columns: list[str] | None = None,
        paired: bool = False,
        do_association: bool = False,
    ) -> WorkflowState:
        """Run the statistical test workflow on the provided data.

        Args:
            data: Input data for analysis.
            target_columns: Columns to analyze. If None, uses all columns.
            paired: Whether data represents paired measurements.
            do_association: Whether to perform association tests.

        Returns:
            Final workflow state with results.
        """
        logger.info('Starting workflow execution')
        logger.info(f'Data shape: {data.shape if hasattr(data, "shape") else len(data)}')

        # Create initial state
        initial_state = create_initial_state(
            df=data,
            target_columns=target_columns,
            paired=paired,
            do_association=do_association,
        )

        # Run workflow
        try:
            final_state = None
            for state_update in self.graph.stream(initial_state):
                # Extract the actual state from the stream
                if isinstance(state_update, dict) and '__end__' not in state_update:
                    # Get the first (and typically only) value from the dict
                    final_state = list(state_update.values())[0]

            if final_state is None:
                final_state = initial_state

            logger.info('Workflow execution completed successfully')
            return final_state

        except Exception as e:
            logger.error(f'Workflow execution failed: {e}')
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

            logger.info(f'Saved workflow diagram to {output_path}')
            return md
        except Exception as e:
            logger.error(f'Failed to generate workflow visualization: {e}')
            raise


def run_workflow(
    data: pd.DataFrame | pd.Series,
    target_columns: list[str] | None = None,
    paired: bool = False,
    do_association: bool = False,
    config: Config | None = None,
) -> WorkflowState:
    """Convenience function to run the workflow.

    Args:
        data: Input data for analysis.
        target_columns: Columns to analyze. If None, uses all columns.
        paired: Whether data represents paired measurements.
        do_association: Whether to perform association tests.
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
    )


if __name__ == '__main__':
    import numpy as np

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
    print(f'Number of tests performed: {len(result.probabilities)}')
    print(f'Tests: {list(result.probabilities.keys())}')
    print('\n=== Results Messages ===')
    for i, msg in enumerate(result.results, 1):
        print(f'\n--- Message {i} ---')
        print(msg.content)
