"""Tests that CORS never pairs a wildcard origin with credentials.

The API sends ``allow_credentials=True``. Combined with a reflected/wildcard origin,
that lets any site issue credentialed cross-origin requests. The app previously set
``allow_origin_regex='.*'`` whenever ``ENVIRONMENT == 'development'`` — which is the
default value — so the permissive path was the one you got by forgetting to configure
anything.
"""

from __future__ import annotations

import pathlib

import pytest

from config.settings import Settings


def test_cors_has_no_wildcard_regex() -> None:
    """The app must not reflect arbitrary origins; it sends credentials."""
    source = pathlib.Path('statmate/api/main.py').read_text()
    assert 'allow_origin_regex' not in source, 'wildcard origin regex reintroduced'


def test_wildcard_cors_origin_is_rejected() -> None:
    """Configuring "*" as an allowed origin fails fast."""
    with pytest.raises(ValueError, match='must not contain'):
        Settings(CORS_ORIGINS=['*'])


def test_explicit_origins_are_accepted() -> None:
    """A normal explicit origin list still validates."""
    settings = Settings(CORS_ORIGINS=['http://localhost:3000'])
    assert settings.CORS_ORIGINS == ['http://localhost:3000']
