# StatMate Module Organization

## Overview

All foundational modules have been organized into the `statmate.core` package for better structure and maintainability.

## Directory Structure

```
statmate/
├── core/                           # Core foundational modules
│   ├── __init__.py                 # Exports all core functionality
│   ├── config.py                   # Configuration management
│   ├── exceptions.py               # Custom exception hierarchy
│   ├── logging_config.py           # Logging setup and configuration
│   ├── base_interfaces.py          # Abstract base classes and protocols
│   └── validation.py               # Input validation utilities
├── agents/                         # AI agent implementations
├── statistical_core/               # Statistical test implementations
└── workflow/                       # Workflow orchestration
```

## Module Purposes

### `statmate.core.config`
**Purpose:** Centralized configuration management

**Key Components:**
- `ModelConfig` - AI model settings (model name, temperature, etc.)
- `StatisticalTestConfig` - Statistical test defaults (alpha, thresholds)
- `WorkflowConfig` - Workflow execution settings
- `LoggingConfig` - Logging configuration
- `Config` - Main configuration class combining all configs
- `default_config` - Global default configuration instance
- Constants: `NodeName`, `DataType`, `TransformationType`

**Usage:**
```python
from statmate.core import Config, default_config, ModelConfig

# Use default config
alpha = default_config.statistical.default_alpha

# Create custom config
config = Config(
    model=ModelConfig(model_name='gpt-4-turbo'),
    statistical=StatisticalTestConfig(default_alpha=0.01)
)
```

### `statmate.core.exceptions`
**Purpose:** Custom exception hierarchy for specific error scenarios

**Key Components:**
- `StatMateError` - Base exception for all StatMate errors
- `DataValidationError` - Data validation failures
  - `InsufficientDataError`
  - `InvalidDataShapeError`
  - `MissingDataError`
  - `InvalidDataTypeError`
- `ConfigurationError` - Configuration issues
- `ModelError` - AI model errors
  - `ModelInitializationError`
  - `ModelInferenceError`
- `WorkflowError` - Workflow execution errors
  - `NodeExecutionError`
- `StatisticalTestError` - Statistical test failures
  - `TestAssumptionViolationError`
- `AgentError` - Agent errors
  - `AgentToolError`
- `TransformationError` - Data transformation errors
  - `InvalidTransformationError`

**Usage:**
```python
from statmate.core import DataValidationError, InsufficientDataError

try:
    # Some operation
    pass
except InsufficientDataError as e:
    print(f"Not enough data: {e}")
```

### `statmate.core.logging_config`
**Purpose:** Structured logging setup and management

**Key Components:**
- `setup_logging(config)` - Initialize logging with configuration
- `get_logger(name)` - Get a module-specific logger

**Usage:**
```python
from statmate.core import get_logger, setup_logging, LoggingConfig

# Setup logging
setup_logging(LoggingConfig(level='DEBUG', log_to_file=True))

# Get logger for your module
logger = get_logger(__name__)
logger.info('Processing data...')
```

### `statmate.core.base_interfaces`
**Purpose:** Abstract base classes and protocols for consistent behavior

**Key Components:**
- `StatisticalTest` (Protocol) - Contract for statistical test functions
- `DataTransformer` (ABC) - Base class for data transformations
- `AgentDependencies` - Base for agent dependency models
- `AgentResult` - Base for agent result models
- `StatMateAgent` (ABC) - Base class for agents
- `WorkflowNode` (ABC) - Base class for workflow nodes
- `DataValidator` (ABC) - Base class for data validators

**Usage:**
```python
from statmate.core.base_interfaces import DataTransformer

class MyTransformer(DataTransformer):
    def transform(self, data, **kwargs):
        # Implementation
        pass
    
    def validate_parameters(self, **kwargs):
        # Implementation
        pass
```

### `statmate.core.validation`
**Purpose:** Comprehensive input validation utilities

**Key Components:**
- `validate_array_not_empty()` - Check array is not empty
- `validate_minimum_sample_size()` - Ensure sufficient data
- `validate_no_missing_values()` - Check for NaN/null values
- `validate_numeric_data()` - Verify data is numeric
- `validate_same_length()` - Check arrays have same length
- `validate_dataframe_columns()` - Verify required columns exist
- `validate_categorical_data()` - Validate categorical data
- `validate_contingency_table()` - Validate contingency tables
- `validate_paired_data()` - Validate paired test data
- `validate_independent_samples()` - Validate independent samples
- `validate_alpha()` - Validate significance level
- `validate_test_parameters()` - Validate test parameters

**Usage:**
```python
from statmate.core.validation import validate_paired_data, validate_alpha

# Validate inputs
validate_alpha(0.05)
validate_paired_data(data1, data2, alpha=0.05)
```

## Import Patterns

### Recommended: Import from `statmate.core`

```python
# Import commonly used components
from statmate.core import (
    Config,
    default_config,
    get_logger,
    DataValidationError,
    InsufficientDataError,
    NodeName,
)
```

### Alternative: Import from specific modules

```python
# Import from specific submodules if needed
from statmate.core.config import Config, ModelConfig
from statmate.core.exceptions import DataValidationError
from statmate.core.logging_config import get_logger
from statmate.core.validation import validate_paired_data
```

## Benefits of This Organization

1. **Clear Separation of Concerns**
   - Core functionality is separate from application logic
   - Easy to find foundational components

2. **Better Namespace Management**
   - `statmate.core` clearly indicates foundational modules
   - Avoids cluttering the top-level namespace

3. **Improved Discoverability**
   - All core exports available through `statmate.core`
   - IDE autocomplete works better

4. **Easier Maintenance**
   - Related modules grouped together
   - Changes to core functionality isolated

5. **Professional Structure**
   - Follows common Python package patterns
   - Similar to frameworks like Django, Flask

## Migration Guide

### Old Import Patterns (Before Reorganization)

```python
from statmate.config import Config, default_config
from statmate.exceptions import DataValidationError
from statmate.logging_config import get_logger
from statmate.validation import validate_paired_data
from statmate.base_interfaces import StatisticalTest
```

### New Import Patterns (After Reorganization)

```python
# Option 1: Import from statmate.core (recommended)
from statmate.core import (
    Config,
    default_config,
    DataValidationError,
    get_logger,
    validate_paired_data,
    StatisticalTest,
)

# Option 2: Import from specific submodules
from statmate.core.config import Config, default_config
from statmate.core.exceptions import DataValidationError
from statmate.core.logging_config import get_logger
from statmate.core.validation import validate_paired_data
from statmate.core.base_interfaces import StatisticalTest
```

## Updated Module Map

| Old Location                  | New Location                       | Purpose       |
| ----------------------------- | ---------------------------------- | ------------- |
| `statmate/config.py`          | `statmate/core/config.py`          | Configuration |
| `statmate/exceptions.py`      | `statmate/core/exceptions.py`      | Exceptions    |
| `statmate/logging_config.py`  | `statmate/core/logging_config.py`  | Logging       |
| `statmate/validation.py`      | `statmate/core/validation.py`      | Validation    |
| `statmate/base_interfaces.py` | `statmate/core/base_interfaces.py` | Interfaces    |

## Testing

All core modules can be tested independently:

```bash
# Test configuration
python -m pytest tests/core/test_config.py

# Test exceptions
python -m pytest tests/core/test_exceptions.py

# Test validation
python -m pytest tests/core/test_validation.py
```

## API Stability

The `statmate.core` public API (exports in `__init__.py`) is considered stable. Internal implementation details in submodules may change, but the public API will maintain backward compatibility.

## Future Enhancements

Potential additions to `statmate.core`:

1. `statmate.core.metrics` - Performance metrics and monitoring
2. `statmate.core.caching` - Result caching utilities
3. `statmate.core.serialization` - Data serialization/deserialization
4. `statmate.core.hooks` - Extension hooks and plugins

## Conclusion

The reorganization into `statmate.core` provides a clean, professional structure that improves maintainability and makes the codebase easier to navigate. All foundational functionality is now logically grouped and easily accessible through a single import point.

