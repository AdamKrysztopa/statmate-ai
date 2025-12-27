"""Lightweight perf smoke test for visualization/export pipeline.

This avoids LLM calls and focuses on data loading + plotting, which matches
what the frontend consumes next to p-values/effect sizes.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from statmate.api.services.export_service import ExportService
from statmate.api.services.visualization_service import VisualizationService


def run(dataset_path: Path) -> None:
    start = time.perf_counter()
    df = pd.read_csv(dataset_path)
    loaded = time.perf_counter()

    viz = VisualizationService.generate_visualizations(df, limit=6)
    viz_done = time.perf_counter()

    html = ExportService.render_html_report(
        {
            'dataset_name': dataset_path.name,
            'summary': 'Perf smoke run',
            'probabilities': {'placeholder-test': 0.05},
            'effect_sizes': viz.get('effect_sizes'),
            'plots': viz.get('plots'),
            'decision_steps': [],
        }
    )
    try:
        pdf_bytes = ExportService.render_pdf(html)
        pdf_done = time.perf_counter()
        print(f'Loaded {len(df):,} rows from {dataset_path} in {(loaded - start):.3f}s')
        print(f'Generated {len(viz.get(\"plots\", []))} plots + effect sizes in {(viz_done - loaded):.3f}s')
        print(f'Rendered PDF ({len(pdf_bytes)} bytes) in {(pdf_done - viz_done):.3f}s')
        print(f'Total runtime: {(pdf_done - start):.3f}s')
    except RuntimeError as exc:
        print('PDF render skipped:', exc)
        print(f'Loaded {len(df):,} rows from {dataset_path} in {(loaded - start):.3f}s')
        print(f'Generated {len(viz.get(\"plots\", []))} plots + effect sizes in {(viz_done - loaded):.3f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perf smoke test for StatMate visualization/export stack')
    parser.add_argument(
        '--dataset',
        type=Path,
        default=Path('frontend/public/samples/clinical_trial_sample.csv'),
        help='Path to CSV dataset to exercise',
    )
    args = parser.parse_args()
    if not args.dataset.exists():
        raise SystemExit(f'Dataset not found: {args.dataset}')
    run(args.dataset)
