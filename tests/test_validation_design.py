import pandas as pd

from statmate.core.validation import infer_statistical_design, validate_statistical_design


def test_infer_statistical_design_detects_paired_overlap():
    df = pd.DataFrame(
        {
            'subject_id': [1, 1, 2, 2],
            'group': ['A', 'B', 'A', 'B'],
            'value': [1, 2, 3, 4],
        }
    )

    design, summary = infer_statistical_design(df)

    assert design.is_paired
    assert design.design_type == 'paired'
    assert design.grouping_variable == 'group'
    assert design.subject_id_column == 'subject_id'
    assert summary['overlap_summary']['shared_ids_across_groups'] == 2


def test_infer_statistical_design_independent_when_no_overlap():
    df = pd.DataFrame(
        {
            'subject_id': [1, 2, 3, 4],
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4],
        }
    )

    design, summary = infer_statistical_design(df)

    assert design.design_type == 'independent'
    assert not design.is_paired
    assert summary['overlap_summary']['shared_ids_across_groups'] == 0


def test_infer_statistical_design_mixed_when_overlap_and_repeats():
    df = pd.DataFrame(
        {
            'subject_id': [1, 1, 1, 2, 2, 3, 3],
            'group': ['A', 'A', 'B', 'B', 'B', 'A', 'B'],
            'value': [0, 1, 2, 3, 4, 5, 6],
        }
    )

    design, summary = infer_statistical_design(df)

    assert design.design_type == 'mixed'
    assert design.is_paired
    assert summary['overlap_summary']['shared_ids_across_groups'] >= 1
    assert summary['overlap_summary']['repeated_within_group'] > 0
    assert design.comparison_matrix.get('comparisons')


def test_validate_statistical_design_wide_paired():
    df = pd.DataFrame(
        {'before_treatment': [1, 2, 3], 'after_treatment': [2, 3, 4]},
    )

    design = validate_statistical_design(df, dependent_var=['before_treatment', 'after_treatment'])

    assert design.is_paired
    assert design.design_type == 'paired'
    rationale = (design.rationale or '').lower()
    assert 'wide-format' in rationale
    assert 'pair' in rationale


def test_validate_statistical_design_independent_groups():
    df = pd.DataFrame({'value': [1, 2, 3, 4], 'group': ['A', 'A', 'B', 'B']})

    design = validate_statistical_design(df, dependent_var=['value'], group_var='group')

    assert not design.is_paired
    assert design.design_type == 'independent'
    assert design.suggested_groups == ['A', 'B']
