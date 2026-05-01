"""Focused tests for reviewer fail-soft medical-reporting flags."""

# ruff: noqa: D103, S101, Q000

from statmate.agents.reviewer_agent import (
    ReviewerDeps,
    ReviewerEvidence,
    ReviewerResult,
    apply_reviewer_informational_flags,
    collect_reviewer_informational_flags,
)
from statmate.agents.summarizer_agent import Finding


def _base_reviewer_result(*, approved: bool, adjusted_summary: str) -> ReviewerResult:
    return ReviewerResult(
        approved=approved,
        adjusted_summary=adjusted_summary,
        hallucination_flags=[],
        risk_score=0.05,
        notes="No unsupported claims were identified.",
    )


def test_missing_findings_are_flagged_without_forcing_rejection() -> None:
    summary = (
        "Summary:\nThe independent_t_test was completed without a statistically significant difference.\n\n"
        "Recommendations:\nInterpret the result in context.\n\n"
        "Performed Tests:\nindependent_t_test"
    )
    deps = ReviewerDeps(
        summary=summary,
        results=[],
        probabilities={"independent_t_test": 0.08},
        performed_tests=["independent_t_test"],
        findings=[],
        result_context=[
            ReviewerEvidence(
                test_name="independent_t_test",
                effect_size_type="hedges_g",
                confidence_interval=[-0.1, 0.4],
                p_value=0.08,
            )
        ],
    )

    merged = apply_reviewer_informational_flags(
        _base_reviewer_result(
            approved=False,
            adjusted_summary="The independent_t_test was completed without a statistically significant difference.",
        ),
        deps,
    )

    assert merged.missing_structure_flags == ["independent_t_test"]
    assert merged.approved is True


def test_informational_flags_do_not_override_substantive_rejection() -> None:
    summary = (
        "Summary:\nThe independent_t_test did not show a statistically significant difference.\n\n"
        "Recommendations:\nInterpret the result in context.\n\n"
        "Performed Tests:\nindependent_t_test"
    )
    deps = ReviewerDeps(
        summary=summary,
        results=[],
        probabilities={"independent_t_test": 0.08},
        performed_tests=["independent_t_test"],
        findings=[],
        result_context=[
            ReviewerEvidence(
                test_name="independent_t_test",
                effect_size_type="hedges_g",
                confidence_interval=[-0.1, 0.4],
                p_value=0.08,
            )
        ],
    )

    merged = apply_reviewer_informational_flags(
        ReviewerResult(
            approved=False,
            adjusted_summary="The independent_t_test did not show a statistically significant difference.",
            hallucination_flags=[],
            risk_score=0.7,
            notes="Evidence is insufficient to verify the reported claim.",
        ),
        deps,
    )

    assert merged.missing_structure_flags == ["independent_t_test"]
    assert merged.approved is False


def test_low_risk_substantive_rejection_stays_rejected() -> None:
    summary = (
        "Summary:\nThe independent_t_test did not show a statistically significant difference.\n\n"
        "Recommendations:\nInterpret the result in context.\n\n"
        "Performed Tests:\nindependent_t_test"
    )
    deps = ReviewerDeps(
        summary=summary,
        results=[],
        probabilities={"independent_t_test": 0.08},
        performed_tests=["independent_t_test"],
        findings=[],
        result_context=[
            ReviewerEvidence(
                test_name="independent_t_test",
                effect_size_type="hedges_g",
                confidence_interval=[-0.1, 0.4],
                p_value=0.08,
            )
        ],
    )

    merged = apply_reviewer_informational_flags(
        ReviewerResult(
            approved=False,
            adjusted_summary="The independent_t_test did not show a statistically significant difference.",
            hallucination_flags=[],
            risk_score=0.1,
            notes="The evidence does not support the reported claim.",
        ),
        deps,
    )

    assert merged.missing_structure_flags == ["independent_t_test"]
    assert merged.approved is False


def test_missing_effect_size_and_ci_are_flagged_for_supported_tests() -> None:
    deps = ReviewerDeps(
        summary="Summary: a t-test was completed.",
        results=[],
        probabilities={"welch_t_test": 0.03},
        performed_tests=["welch_t_test"],
        findings=[
            Finding(
                finding="The welch_t_test found higher scores in group A.",
                evidence="Welch t = 2.19, p = 0.03.",
                caveat=None,
            )
        ],
        result_context=[
            ReviewerEvidence(test_name="welch_t_test", effect_size_type=None, confidence_interval=None, p_value=0.03)
        ],
    )

    flags = collect_reviewer_informational_flags(deps)

    assert flags.missing_effect_size_flags == ["welch_t_test"]
    assert flags.missing_ci_flags == ["welch_t_test"]


def test_multiplicity_warning_uses_bonferroni_for_multiple_comparisons() -> None:
    deps = ReviewerDeps(
        summary="Summary: two pairwise comparisons were completed.",
        results=[],
        probabilities={"t_test": 0.03, "welch_t": 0.04},
        performed_tests=["t_test", "welch_t"],
        findings=[
            Finding(
                finding="The t_test showed a higher mean in group A.",
                evidence="t = 2.21, p = 0.03.",
                caveat=None,
            ),
            Finding(
                finding="The welch_t showed a higher mean in group B.",
                evidence="Welch t = 2.06, p = 0.04.",
                caveat=None,
            ),
        ],
        result_context=[
            ReviewerEvidence(
                test_name="t_test", effect_size_type="hedges_g", confidence_interval=[0.1, 0.6], p_value=0.03
            ),
            ReviewerEvidence(
                test_name="welch_t", effect_size_type="hedges_g", confidence_interval=[0.0, 0.5], p_value=0.04
            ),
        ],
    )

    flags = collect_reviewer_informational_flags(deps)

    assert flags.multiplicity_warning is not None
    assert "Bonferroni" in flags.multiplicity_warning
    assert "0.0250" in flags.multiplicity_warning


def test_diagnostic_probability_keys_do_not_create_information_only_advisories() -> None:
    deps = ReviewerDeps(
        summary="Summary: welch_t_test was completed.",
        results=[],
        probabilities={
            "welch_t_test": 0.03,
            "shapiro_group1": 0.42,
            "levene": 0.31,
        },
        performed_tests=["welch_t_test", "shapiro_group1", "levene"],
        findings=[
            Finding(
                finding="The welch_t_test found higher scores in group A.",
                evidence="Welch t = 2.19, p = 0.03.",
                caveat=None,
            )
        ],
        result_context=[
            ReviewerEvidence(
                test_name="welch_t_test",
                effect_size_type="hedges_g",
                confidence_interval=[0.1, 0.6],
                p_value=0.03,
            )
        ],
    )

    flags = collect_reviewer_informational_flags(deps)

    assert flags.missing_structure_flags == []
    assert flags.multiplicity_warning is None

    merged = apply_reviewer_informational_flags(
        _base_reviewer_result(
            approved=False,
            adjusted_summary="welch_t_test was completed.",
        ),
        deps,
    )

    assert merged.approved is False
    assert merged.missing_structure_flags == []
    assert merged.multiplicity_warning is None
