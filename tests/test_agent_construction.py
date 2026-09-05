"""Regression tests for agent construction and report-export escaping.

These guard the two failure classes that the pydantic-ai 2.x upgrade exposed:

1. Every agent factory is constructed lazily, so an incompatible ``Agent(...)``
   keyword (``result_type`` was removed in 2.x) raised only at call time and no
   existing test reached it. These tests build every agent for real.
2. Report HTML interpolates dataset names and LLM-authored text into quoted
   attributes, where unescaped quotes allow attribute/CSS injection into the
   document handed to WeasyPrint.
"""

from __future__ import annotations

import base64
from html.parser import HTMLParser
from typing import Any

import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings

from statmate.agents.agent_builder import build_stat_test_agent
from statmate.api.services.export_service import ExportService


def _agent_factories() -> list[tuple[str, Any]]:
    """Return (name, zero-argument builder) for each agent factory under test."""
    from statmate.agents.auxiliary_agents import get_assess_design_study_agent
    from statmate.agents.initial_insights_agent import build_initial_insights_agent
    from statmate.agents.normality_agent import meta_normality_agent
    from statmate.agents.reviewer_agent import get_reviewer_agent
    from statmate.agents.summarizer_agent import get_summariser_agent
    from statmate.workflow.initialization.column_role_agent import build_column_role_agent

    model = TestModel()
    settings = ModelSettings()
    return [
        ('assess_design', lambda: get_assess_design_study_agent(model, settings)),
        ('initial_insights', lambda: build_initial_insights_agent(model, system_prompt='test prompt')),
        ('meta_normality', lambda: meta_normality_agent(model)),
        ('reviewer', lambda: get_reviewer_agent(model, settings)),
        ('summariser', lambda: get_summariser_agent(model, settings)),
        ('column_role', lambda: build_column_role_agent(model)),
    ]


@pytest.mark.parametrize('name,factory', _agent_factories(), ids=lambda v: v if isinstance(v, str) else '')
def test_agent_factory_constructs(name: str, factory: Any) -> None:
    """Each agent factory builds against the installed pydantic-ai version."""
    agent = factory()
    assert agent is not None, f'{name} factory returned nothing'


def test_build_stat_test_agent_constructs() -> None:
    """The generic statistical-agent builder constructs with a test model."""
    from statmate.statistical_core.normality import shapiro_wilk_test

    agent = build_stat_test_agent(
        model=TestModel(),
        test_name='shapiro_wilk',
        test_function=shapiro_wilk_test,
    )
    assert agent is not None


def test_agent_run_result_field_is_output() -> None:
    """Run results expose ``.output``; ``.data`` was removed in pydantic-ai 2.x."""
    import dataclasses

    from pydantic_ai.run import AgentRunResult

    names = {f.name for f in dataclasses.fields(AgentRunResult)}
    assert 'output' in names
    assert 'data' not in names


class _AttrCollector(HTMLParser):
    """Collect attributes that would indicate a successful markup breakout."""

    def __init__(self) -> None:
        super().__init__()
        self.injected: list[tuple[str, str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if key.startswith('on') or (key == 'style' and value and 'url(' in value):
                self.injected.append((tag, key, value))


def test_report_html_neutralises_attribute_injection() -> None:
    """Quote-bearing dataset/LLM text cannot break out of quoted attributes."""
    payload = "x' style='background:url(http://evil/x)' onload='alert(1)"
    document = ExportService.render_html_report(
        {
            'dataset_name': '<img src=x onerror=alert(1)>"evil"',
            'summary': 'it\'s <b>bold</b> & "quoted"',
            'probabilities': {"col' onmouseover='x": 0.01},
            'plots': [
                {
                    'title': payload,
                    'description': payload,
                    'content_type': 'image/png" style="x',
                    'image_base64': 'AAA" onerror="y',
                }
            ],
            'workflow_graph': {'assets': {'alt': payload, 'svg_base64': "BBB' onerror='z"}},
            'decision_steps': [
                {'step': payload, 'detail': payload, 'timestamp': 't', 'p_value': None, 'progress_pct': 10}
            ],
        }
    )
    collector = _AttrCollector()
    collector.feed(document)
    assert collector.injected == [], f'markup breakout: {collector.injected}'


def test_html_safe_escapes_both_quote_styles() -> None:
    """``_html_safe`` must cover attribute contexts, not just text nodes."""
    escaped = ExportService._html_safe('a"b\'c<d>e&f')
    for raw in ('"', "'", '<', '>'):
        assert raw not in escaped, f'{raw!r} left unescaped in {escaped!r}'


def test_image_data_uri_rejects_malformed_payload() -> None:
    """Non-base64 payloads are dropped rather than emitted into the document."""
    assert ExportService._image_data_uri('image/png', 'AAA" onerror="y') is None
    assert ExportService._image_data_uri('image/png', None) is None


def test_image_data_uri_forces_known_mime() -> None:
    """An unexpected MIME type falls back to image/png instead of being trusted."""
    uri = ExportService._image_data_uri('image/png" style="x', base64.b64encode(b'abc').decode())
    assert uri is not None
    assert uri.startswith('data:image/png;base64,')


def test_pdf_renderer_blocks_network_urls() -> None:
    """WeasyPrint may not fetch remote resources during report rendering."""
    with pytest.raises(ValueError, match='Blocked non-data URL'):
        ExportService._blocked_url_fetcher('http://evil.example/x.png')
    with pytest.raises(ValueError, match='Blocked non-data URL'):
        ExportService._blocked_url_fetcher('file:///etc/passwd')


def test_pdf_render_still_works() -> None:
    """The blocked fetcher does not break ordinary rendering."""
    doc = ExportService.render_html_report({'dataset_name': 'D', 'summary': 's'})
    assert ExportService.render_pdf(doc)
