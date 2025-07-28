"""Hacker_News_Source engine module.

This module provides hacker news source functionality for the Haive framework.

Classes:
    HackerNewsSource: HackerNewsSource implementation.

Functions:
    validate_url: Validate Url functionality.
"""

from typing import Any

from pydantic import Field, HttpUrl

from haive.core.engine.loaders.sources.remote.base import URLSource
from haive.core.engine.loaders.sources.types import SourceType


class HackerNewsSource(URLSource):
    """A source that is a hacker news post."""

    source_type: SourceType = Field(default=SourceType.HACKER_NEWS)
    url: HttpUrl = Field(description="The url of the hacker news post.")

    @field_validatorvalidate_url
    @classmethod
    def validate_url(cls, v) -> Any:
        if not v.startswith("https://news.ycombinator.com/"):
            raise ValueError(f"Invalid url: {v}")
        return v
