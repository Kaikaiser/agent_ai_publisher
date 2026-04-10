from fastapi import Depends, HTTPException, status

from app.services.providers import OpenAICompatibleProvider


def get_provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider()


def _wrap_provider_error(func):
    # Provider misconfiguration should surface as a clean HTTP error instead of a raw stack trace.
    try:
        return func()
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))


def get_llm(provider: OpenAICompatibleProvider = Depends(get_provider)):
    return _wrap_provider_error(provider.get_llm)


def get_vision_llm(provider: OpenAICompatibleProvider = Depends(get_provider)):
    return _wrap_provider_error(provider.get_vision_llm)


def get_embeddings(provider: OpenAICompatibleProvider = Depends(get_provider)):
    return _wrap_provider_error(provider.get_embeddings)