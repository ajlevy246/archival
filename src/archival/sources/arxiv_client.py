
"""HTTP client for retrieving ArXiv OAI-PMH ListRecords pages."""

import time
from dataclasses import dataclass
from typing import Self

import httpx

from archival.sources.arxiv_oai import (
    ParsedHarvestPage,
    parse_list_records,
)

OAI_ENDPOINT = "https://oaipmh.arxiv.org/oai"


@dataclass
class ArxivOAIClient:
    """Fetch and parse pages from the ArXiv OAI-PMH endpoint."""

    endpoint: str = OAI_ENDPOINT
    timeout_seconds: float = 60.0
    min_request_interval: float = 3.0

    def __post_init__(self) -> None:
        """Initialize the HTTP client and request-rate limiter."""
        self._http = httpx.Client(
            timeout=self.timeout_seconds,
            headers={"User-Agent": "Archival/0.1 (ArXiv metadata harvester)"},
        )
        self._last_request_at = 0.0

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> Self:
        """Support use as a context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the client when leaving a context."""
        self.close()

    def fetch_page(
        self,
        resumption_token: str | None = None,
    ) -> ParsedHarvestPage:
        """Fetch and parse one page, starting a harvest or resuming it."""
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

        if resumption_token:
            params = {
                "verb": "ListRecords",
                "resumptionToken": resumption_token,
            }
        else:
            params = {
                "verb": "ListRecords",
                "metadataPrefix": "arXiv",
            }

        response = self._http.get(self.endpoint, params=params)
        self._last_request_at = time.monotonic()
        response.raise_for_status()

        return parse_list_records(response.content)