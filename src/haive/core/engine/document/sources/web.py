import re
from typing import Any, ClassVar, Self
from urllib.parse import parse_qs, urlparse

from pydantic import Field, HttpUrl, computed_field, model_validator

from haive.core.engine.loaders.sources.base import BaseSource
from haive.core.engine.loaders.sources.types import SourceType


class WebSource(BaseSource):
    """Base class for all web-based sources."""

    url: HttpUrl = Field(description="URL of the resource")
    headers: dict[str, str] = Field(default_factory=dict, description="HTTP headers")
    timeout: int = Field(default=30, description="Timeout in seconds")
    verify_ssl: bool = Field(
        default=True, description="Whether to verify SSL certificates"
    )
    DOMAIN_PATTERNS: ClassVar[dict[str, SourceType]] = {
        "github\\.com": SourceType.GITHUB,
        "arxiv\\.org": SourceType.ARXIV,
        "reddit\\.com": SourceType.REDDIT,
        "twitter\\.com": SourceType.TWITTER,
        "youtube\\.com": SourceType.YOUTUBE,
        "youtu\\.be": SourceType.YOUTUBE,
        "wikipedia\\.org": SourceType.WIKIPEDIA,
        "news\\.ycombinator\\.com": SourceType.HACKER_NEWS,
        "collegeconfidential\\.com": SourceType.COLLEGE_CONFIDENTIAL,
    }

    @model_validator(mode="after")
    def validate_source_type(self) -> Self:
        """Automatically set more specific source type based on URL domain."""
        url_str = str(self.url)
        parsed = urlparse(url_str)
        domain = parsed.netloc
        for pattern, source_type in self.DOMAIN_PATTERNS.items():
            if re.search(pattern, domain, re.IGNORECASE):
                self.source_type = source_type
                break
        return self

    def get_source_value(self) -> HttpUrl:
        """Get the URL as the source value."""
        return self.url

    def validate(self) -> bool:
        """Basic validation of URL format.
        Note: This doesn't check if the URL is accessible.
        """
        return True

    @computed_field
    def domain(self) -> str:
        """Get the domain from the URL."""
        parsed = urlparse(str(self.url))
        return parsed.netloc

    @computed_field
    def scheme(self) -> str:
        """Get the scheme from the URL."""
        parsed = urlparse(str(self.url))
        return parsed.scheme

    @computed_field
    def path(self) -> str:
        """Get the path from the URL."""
        parsed = urlparse(str(self.url))
        return parsed.path

    @computed_field
    def query_params(self) -> dict[str, list[str]]:
        """Get the query parameters from the URL."""
        parsed = urlparse(str(self.url))
        return parse_qs(parsed.query)

    def get_metadata(self) -> dict[str, Any]:
        """Get URL metadata."""
        metadata = super().get_metadata()
        metadata.update(
            {
                "url": str(self.url),
                "domain": self.domain,
                "scheme": self.scheme,
                "path": self.path,
                "headers": {
                    k: v
                    for k, v in self.headers.items()
                    if k.lower() != "authorization"
                },
            }
        )
        return metadata

    @classmethod
    def from_url(cls, url: str | HttpUrl, **kwargs) -> "WebSource":
        """Create a WebSource from a URL."""
        if "name" not in kwargs:
            parsed = urlparse(str(url))
            kwargs["name"] = parsed.netloc
        return cls(url=url, **kwargs)


class ApiSource(WebSource):
    """Source for API endpoints."""

    source_type: SourceType = Field(default=SourceType.API)
    method: str = Field(default="GET", description="HTTP method")
    params: dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    data: dict[str, Any] | None = Field(default=None, description="Request data")
    auth: dict[str, str] | None = Field(
        default=None, description="Authentication credentials"
    )

    def get_metadata(self) -> dict[str, Any]:
        """Get API metadata."""
        metadata = super().get_metadata()
        metadata.update({"method": self.method, "params": self.params})
        return metadata
