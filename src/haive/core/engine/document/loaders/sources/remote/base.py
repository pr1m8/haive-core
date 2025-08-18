from typing import Any

from pydantic import Field, HttpUrl, field_validator

from haive.core.engine.document.loaders.sources.base import BaseSource
from haive.core.engine.document.loaders.sources.types import SourceType


class URLSource(BaseSource):
    """A source that is a remote file."""

    source_type: SourceType = Field(default=SourceType.URL)
    url: HttpUrl = Field(description="The url of the remote file.")
    url_prefix: str | None = Field(default="", description="The prefix of the url.")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v) -> Any:
        """Validate Url.

        Args:
            v: [TODO: Add description]

        Returns:
            [TODO: Add return description]
        """
        if not v.is_valid():
            raise ValueError(f"Invalid url: {v}")
        return v

    @property
    def source(self) -> HttpUrl:
        """The source of the remote file."""
        return self.url

    @classmethod
    def from_url(cls, url: HttpUrl | str) -> "URLSource":
        """Create a URLSource from a url."""
        if isinstance(url, str):
            url = HttpUrl(url)
        return cls(url=url)
