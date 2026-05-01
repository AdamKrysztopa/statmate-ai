"""Pydantic models for model configuration API."""

from pydantic import BaseModel, Field


class ModelInfoResponse(BaseModel):
    """Response model for model information."""

    name: str = Field(..., description='Model name (e.g., gpt-4o)')
    provider: str = Field(..., description='Provider name (e.g., openai)')
    display_name: str = Field(..., description='Human-readable display name')
    description: str = Field(..., description='Model description')
    context_window: int = Field(..., description='Context window size')
    supports_tools: bool = Field(..., description='Whether model supports tool calling')
    capabilities: list[str] = Field(..., description='List of model capabilities')


class AvailableModelsResponse(BaseModel):
    """Response model for available models."""

    models: list[ModelInfoResponse] = Field(..., description='List of available models')
    default_model: str | None = Field(None, description='Current default model name')
    default_provider: str | None = Field(None, description='Current default provider')


class ModelSelectionRequest(BaseModel):
    """Request model for selecting a model for analysis."""

    model_name: str | None = Field(None, description='Model name to use (optional)')
    provider: str | None = Field(None, description='Provider to use (optional)')


class CurrentModelResponse(BaseModel):
    """Response model for current model configuration."""

    model_name: str = Field(..., description='Current default model name')
    provider: str = Field(..., description='Current default provider')
    temperature: float = Field(..., description='Model temperature')
    require_reasoning_for_tools: bool = Field(..., description='Whether reasoning models are required for tools')
    available_providers: list[str] = Field(..., description='List of configured providers')


class ModelValidationResponse(BaseModel):
    """Response model for validating a model/provider combination."""

    requested_model: str = Field(..., description='Requested model name')
    resolved_model: str = Field(..., description='Resolved model name after alias lookup')
    provider: str = Field(..., description='Resolved provider name')
    valid: bool = Field(..., description='Whether the model is valid for the current configuration')
    error: str | None = Field(None, description='Validation error if invalid')
