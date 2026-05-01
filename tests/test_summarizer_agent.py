# tests/test_summarizer_agent.py
"""Unit tests for SummariserResults model structure and __str__ method."""

from statmate.agents.summarizer_agent import SummariserResults


def _base_kwargs() -> dict:
    return {
        'summary': 'The t-test yielded t=1.23, p=0.22.',
        'recommendations': 'Collect more data before drawing conclusions.',
        'performed_tests': ['independent_t_test'],
    }


class TestSummariserResultsPowerInterpretation:
    def test_power_interpretation_stored_when_provided(self) -> None:
        result = SummariserResults(
            **_base_kwargs(),
            power_interpretation='A non-significant result does not confirm the null hypothesis.',
        )
        assert result.power_interpretation is not None
        assert len(result.power_interpretation) > 0

    def test_power_interpretation_defaults_to_none(self) -> None:
        result = SummariserResults(**_base_kwargs())
        assert result.power_interpretation is None

    def test_str_excludes_power_section_when_none(self) -> None:
        result = SummariserResults(**_base_kwargs(), power_interpretation=None)
        assert 'Power Interpretation' not in str(result)

    def test_str_includes_power_section_when_set(self) -> None:
        note = 'A non-significant result does not confirm the null hypothesis.'
        result = SummariserResults(**_base_kwargs(), power_interpretation=note)
        output = str(result)
        assert 'Power Interpretation' in output
        assert note in output

    def test_str_contains_core_sections(self) -> None:
        result = SummariserResults(**_base_kwargs())
        output = str(result)
        assert 'Summary:' in output
        assert 'Recommendations:' in output
        assert 'Performed Tests:' in output
