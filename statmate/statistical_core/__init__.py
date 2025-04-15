"""Statistical Core Module."""

from statmate.statistical_core.base import StatTestResult
from statmate.statistical_core.normality import shapiro_wilk_test

__all__ = [
    'shapiro_wilk_test',
    'StatTestResult',
]
__version__ = '0.1.0'
__author__ = 'MechAI Adam Krysztopa'
