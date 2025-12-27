"""Utilities for exporting analysis results to PDF, DOCX, and CSV."""

from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Any


class ExportService:
    """Build lightweight exports for analyses."""

    @staticmethod
    def _html_safe(text: str | None) -> str:
        return (text or '').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    @classmethod
    def render_html_report(cls, analysis: dict[str, Any]) -> str:
        """Render an HTML report that can be used for PDF/DOCX conversion."""
        summary = cls._html_safe(analysis.get('summary') or analysis.get('results_detail', {}).get('summary'))
        probabilities = analysis.get('probabilities') or {}
        effect_sizes = analysis.get('effect_sizes') or analysis.get('results_detail', {}).get('effect_sizes') or {}
        plots = analysis.get('plots') or analysis.get('results_detail', {}).get('plots') or []
        decision_steps = analysis.get('decision_steps') or []

        table_rows = ''.join(
            f"<tr><td>{cls._html_safe(name)}</td>"
            f"<td>{probabilities[name]:.4f}</td>"
            f"<td>{'Significant' if probabilities[name] < 0.05 else 'Not significant'}</td></tr>"
            for name in probabilities
        )

        effect_rows = ''.join(
            f"<tr><td>{cls._html_safe(name)}</td><td>{value:.3f}</td></tr>" for name, value in effect_sizes.items()
        )

        plot_blocks = ''.join(
            f"""
            <div class="plot">
              <div class="plot-title">{cls._html_safe(plot.get('title') or 'Plot')}</div>
              <img src="data:{plot.get('content_type', 'image/png')};base64,{plot.get('image_base64')}" />
              <div class="plot-meta">{cls._html_safe(plot.get('description') or '')}</div>
            </div>
            """
            for plot in plots
        )

        steps = ''.join(
            f"<li><strong>{cls._html_safe(step.get('step'))}</strong> — {cls._html_safe(step.get('detail'))}</li>"
            for step in decision_steps
        )

        generated_at = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
        dataset_name = analysis.get('dataset_name') or 'Dataset'
        header_title = cls._html_safe(dataset_name)

        return f"""
        <html>
          <head>
            <style>
              body {{ font-family: Arial, sans-serif; color: #0f172a; padding: 24px; }}
              h1 {{ margin-bottom: 6px; }}
              .muted {{ color: #475569; }}
              table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
              th, td {{ border: 1px solid #e2e8f0; padding: 8px; text-align: left; }}
              th {{ background: #f8fafc; }}
              .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }}
              .plot {{ border: 1px solid #e2e8f0; border-radius: 10px; overflow: hidden; }}
              .plot img {{ width: 100%; display: block; }}
              .plot-title {{ font-weight: 700; padding: 8px; background: #f8fafc; }}
              .plot-meta {{ padding: 8px; color: #475569; }}
            </style>
          </head>
          <body>
            <h1>Analysis Report · {header_title}</h1>
            <p class="muted">Generated at {generated_at}</p>
            <h2>Summary</h2>
            <p>{summary or 'No summary available.'}</p>
            <h2>Statistical tests</h2>
            <table>
              <thead><tr><th>Test</th><th>p-value</th><th>Callout</th></tr></thead>
              <tbody>
                {table_rows or '<tr><td colspan="3">No tests reported.</td></tr>'}
              </tbody>
            </table>
            <h2>Effect sizes</h2>
            <table>
              <thead><tr><th>Metric</th><th>Value</th></tr></thead>
              <tbody>
                {effect_rows or '<tr><td colspan="2">No effect sizes computed.</td></tr>'}
              </tbody>
            </table>
            <h2>Plots</h2>
            <div class="plot-grid">
              {plot_blocks or '<div class="muted">No plots generated.</div>'}
            </div>
            <h2>Decision path</h2>
            <ol>{steps or '<li>No steps captured.</li>'}</ol>
          </body>
        </html>
        """

    @staticmethod
    def render_pdf(html: str) -> bytes:
        """Render HTML to PDF using WeasyPrint."""
        try:
            from weasyprint import HTML
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError('WeasyPrint is required for PDF export') from exc
        return HTML(string=html).write_pdf()

    @staticmethod
    def render_docx(analysis: dict[str, Any]) -> bytes:
        """Render a small DOCX using python-docx."""
        try:
            from docx import Document
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError('python-docx is required for DOCX export') from exc

        doc = Document()
        doc.add_heading('StatMate Analysis Report', 0)
        doc.add_paragraph(f"Dataset: {analysis.get('dataset_name') or analysis.get('dataset_id')}")
        doc.add_paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

        summary = analysis.get('summary') or analysis.get('results_detail', {}).get('summary')
        doc.add_heading('Summary', level=1)
        doc.add_paragraph(summary or 'No summary available.')

        probabilities = analysis.get('probabilities') or {}
        if probabilities:
            doc.add_heading('Statistical tests', level=1)
            table = doc.add_table(rows=1, cols=3)
            hdr_cells = table.rows[0].cells
            hdr_cells[0].text = 'Test'
            hdr_cells[1].text = 'p-value'
            hdr_cells[2].text = 'Callout'
            for name, p_val in probabilities.items():
                row_cells = table.add_row().cells
                row_cells[0].text = name
                row_cells[1].text = f'{p_val:.4f}'
                row_cells[2].text = 'Significant' if p_val < 0.05 else 'Not significant'

        effect_sizes = analysis.get('effect_sizes') or analysis.get('results_detail', {}).get('effect_sizes') or {}
        if effect_sizes:
            doc.add_heading('Effect sizes', level=1)
            table = doc.add_table(rows=1, cols=2)
            hdr_cells = table.rows[0].cells
            hdr_cells[0].text = 'Metric'
            hdr_cells[1].text = 'Value'
            for name, val in effect_sizes.items():
                row_cells = table.add_row().cells
                row_cells[0].text = name
                row_cells[1].text = f'{val:.3f}'

        stream = io.BytesIO()
        doc.save(stream)
        stream.seek(0)
        return stream.read()

    @staticmethod
    def render_csv(analysis: dict[str, Any]) -> bytes:
        """Render a compact CSV with probabilities and effect sizes."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['metric', 'p_value', 'effect_size', 'callout'])
        probabilities = analysis.get('probabilities') or {}
        effect_sizes = analysis.get('effect_sizes') or analysis.get('results_detail', {}).get('effect_sizes') or {}
        keys = set(probabilities.keys()) | set(effect_sizes.keys())
        for name in keys:
            p_val = probabilities.get(name)
            effect_val = effect_sizes.get(name)
            callout = ''
            if p_val is not None:
                callout = 'Significant' if p_val < 0.05 else 'Not significant'
            writer.writerow([name, p_val if p_val is not None else '', effect_val if effect_val is not None else '', callout])
        return output.getvalue().encode('utf-8')
