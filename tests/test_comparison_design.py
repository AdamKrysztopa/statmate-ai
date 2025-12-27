import numpy as np
import pytest

from statmate.core.exceptions import DataValidationError
from statmate.statistical_core.comparison import (
    mannwhitneyu_test,
    ttest_ind_test,
    ttest_rel_test,
)


def test_independent_tests_reject_paired_design_flag():
    data1 = np.array([1.0, 2.0, 3.0, 4.0])
    data2 = np.array([1.5, 2.5, 3.5, 4.5])

    with pytest.raises(DataValidationError):
        ttest_ind_test(data1, data2, design_type='paired')

    with pytest.raises(DataValidationError):
        mannwhitneyu_test(data1, data2, design_type='mixed')


def test_paired_tests_reject_independent_design_flag():
    data1 = np.array([1.0, 2.0, 3.0, 4.0])
    data2 = np.array([1.5, 2.5, 3.5, 4.5])

    with pytest.raises(DataValidationError):
        ttest_rel_test(data1, data2, design_type='independent')
