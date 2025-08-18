from pydantic import Field, HttpUrl, field_validator

from haive.core.engine.loaders.sources.base import BaseSource
from haive.core.engine.loaders.sources.types import SourceType


class RemoteSource(BaseSource):
    """A source that is a remote file."""

    source_type: SourceType = Field(
        default=SourceType.REMOTE, description="The type of source."
    )
    url: HttpUrl = Field(description="The url of the remote file.")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v):
        """Validate Url.

        Args:
            v: [TODO: Add description]
        """
        if not v.is_valid():
            raise ValueError(f"Invalid url: {v}")
        return v

    @property
    def source(self) -> HttpUrl:
        """Source.

        Returns:
            [TODO: Add return description]
        """
        return self.url

    @classmethod
    def from_url(cls, url: HttpUrl | str) -> "RemoteSource":
        """From Url.

        Args:
            url: [TODO: Add description]

        Returns:
            [TODO: Add return description]
        """
        if isinstance(url, str):
            url = HttpUrl(url)
        return cls(url=url)
