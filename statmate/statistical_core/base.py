"""Base class for statistical models ouput."""

from typing import Any

from pydantic import BaseModel, Field


class StatTestResult(BaseModel):
    """Statistical test results."""

    test_name: str = Field(description='Name of the performed statistical test')
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
        description='Description of the results basing on null hypothesis and the p-value',
    )
    test_specifics: dict[str, Any] | None = Field(
        description='Test specific parameters, like alpha values, used methods, etc.',
        default=None,
    )

    def __str__(self: 'StatTestResult') -> str:
        p_val_str = f'{self.p_value:.3f}' if isinstance(self.p_value, float) else str(self.p_value)
        alt_str = f'Alternative hypothesis: {self.alternative}\n' if self.alternative else ''
        return (
            f'### Test name: {self.test_name} ###\n'
            f'Null hypothesis: {self.null_hypothesis}\n'
            f'{alt_str}'
            f'p value of {p_val_str} makes: '
            f'{self.statistical_test_results}\n'
        )
