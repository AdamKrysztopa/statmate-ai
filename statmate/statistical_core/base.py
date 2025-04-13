"""Base class for statistical models ouput."""

from typing import Any

from pydantic import BaseModel, Field


class StatTestResult(BaseModel):
    """Statistical test results."""

    test_name: str = Field(description='Name of the preformed statistical test')
    statistics: float | list[float] = Field(
        description='Statistical model statistics values.',
    )
    p_value: float | list[float] = Field(description='Probability value')
    null_hypothesis: str = Field(description='Test detailed null hypothesis.')
    alternative: str | None = Field(
        description='Alternative hypothesis if it is not clear from null',
        default=None,
    )
    statistical_test_results: str = Field(
        description='Description of the results basing on null hypothesis '
        'and the p-value',
    )
    test_specifics: dict[str, Any] | None = Field(
        description='Test speyfic parameters, like alpha values, used methords, etc.',
        default=None,
    )
