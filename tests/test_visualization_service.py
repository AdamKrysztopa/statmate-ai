"""Focused tests for visualization chart captions."""

import pandas as pd

from statmate.api.services.visualization_service import VisualizationService


def test_generate_visualizations_adds_non_empty_captions() -> None:
    df = pd.DataFrame(
        {
            'group': ['control', 'control', 'control', 'treatment', 'treatment', 'treatment'] * 2,
            'age': [50, 52, 49, 60, 61, 59, 51, 53, 50, 62, 64, 63],
            'score': [10, 11, 9, 14, 15, 13, 10, 12, 11, 15, 16, 14],
        }
    )

    payload = VisualizationService.generate_visualizations(df, limit=8)

    plots = payload['plots']
    plot_types = {plot['type'] for plot in plots}

    assert plots
    assert {'histogram', 'box', 'scatter', 'qq', 'bar'}.issubset(plot_types)
    assert all(isinstance(plot.get('caption'), str) and plot['caption'].strip() for plot in plots)
