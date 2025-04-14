"""Statistical Core Module."""

from base import StatTestResult
from normality import shapiro_wilk_test

__all__ = [
    'shapiro_wilk_test',
    'StatTestResult',
]
__version__ = '0.1.0'
__author__ = 'MechAI Adam Krysztopa'
