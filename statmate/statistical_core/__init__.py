"""Statistical Core Module."""

from statmate.statistical_core.anova import (
    anova_one_way_test,
    anova_rm_test,
)
from statmate.statistical_core.base import StatTestResult
from statmate.statistical_core.cathegorical_comparison import (
    chi2_test,
    fisher_exact_test,
)
from statmate.statistical_core.comparison import (
    mannwhitneyu_test,
    ttest_ind_test,
    ttest_rel_test,
    welch_t_test,
    wilcoxon_test,
)
from statmate.statistical_core.normality import (
    anderson_darling_test,
    cramer_von_mises_test,
    dagostino_pearson_test,
    jarque_bera_test,
    ks_normal_test,
    lilliefors_test,
    normality_of_difference,
    shapiro_wilk_test,
)

__all__ = [
    'shapiro_wilk_test',
    'StatTestResult',
    'ttest_rel_test',
    'wilcoxon_test',
    'normality_of_difference',
    'ttest_ind_test',
    'mannwhitneyu_test',
    'welch_t_test',
    'anderson_darling_test',
    'ks_normal_test',
    'dagostino_pearson_test',
    'lilliefors_test',
    'jarque_bera_test',
    'cramer_von_mises_test',
    'anova_one_way_test',
    'anova_rm_test',
    'chi2_test',
    'fisher_exact_test',
]
__version__ = '0.1.0'
__author__ = 'MechAI Adam Krysztopa'
