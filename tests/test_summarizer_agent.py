# tests/test_summarizer_agent.py
"""Unit tests for SummariserResults model structure and __str__ method."""

from statmate.agents.summarizer_agent import Finding, SummariserResults


def _base_kwargs() -> dict[str, object]:
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
        assert result.findings == []

    def test_findings_parse_matches_performed_tests_length(self) -> None:
        result = SummariserResults.model_validate(
            {
                'summary': 'Two tests were completed.',
                'recommendations': 'Review both results in context.',
                'performed_tests': ['welch_t_test', 'pearson_correlation'],
                'power_interpretation': None,
                'findings': [
                    {
                        'finding': 'The welch_t_test found higher scores in the intervention group.',
                        'evidence': 'Welch t = 2.41, p = 0.018.',
                        'caveat': None,
                    },
                    {
                        'finding': 'The pearson_correlation showed a positive association.',
                        'evidence': 'Pearson r = 0.52, p = 0.004.',
                        'caveat': 'The sample size was modest.',
                    },
                ],
            }
        )
        assert len(result.findings) == len(result.performed_tests)
        assert isinstance(result.findings[0], Finding)

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

    def test_str_includes_findings_section_when_present(self) -> None:
        result = SummariserResults(
            **_base_kwargs(),
            findings=[
                Finding(
                    finding='The independent_t_test did not detect a directional difference.',
                    evidence='t = 1.23, p = 0.22.',
                    caveat='The sample size was limited.',
                )
            ],
        )
        output = str(result)
        assert 'Findings:' in output
        assert 'Evidence:' in output
        assert 'Caveat:' in output
