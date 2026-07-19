# ── Custom Exceptions ───────────────────────────────────────────────────────────
"""Exceções customizadas independentes sem imports circulares"""

from fastapi import HTTPException


class THCError(HTTPException):
    """Base exception for THC LLM application"""
    def __init__(self, detail: str, status_code: int = 500):
        super().__init__(status_code=status_code, detail=detail)


class ModelNotFoundError(THCError):
    """Raised when a requested model is not found"""
    def __init__(self, detail: str):
        super().__init__(detail=detail, status_code=400)


class ModelLoadError(THCError):
    """Raised when a model fails to load"""
    pass


class BackendError(THCError):
    """Raised when a backend operation fails"""
    pass


class ConfigurationError(THCError):
    """Raised when configuration is missing or invalid"""
    pass


class APIError(THCError):
    """Raised when external API calls fail"""
    pass


class ImageGenerationError(THCError):
    """Raised when image generation fails"""
    pass


class ChatError(THCError):
    """Raised when chat completion fails"""
    pass
