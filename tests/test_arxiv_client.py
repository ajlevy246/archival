
"""Tests for the ArXiv OAI-PMH HTTP client."""

from unittest.mock import patch

import httpx
import pytest

from archival.sources.arxiv_client import ArxivOAIClient

XML_RESPONSE = b"""\
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <ListRecords>
    <record>
      <header>
        <identifier>oai:arXiv.org:0804.2273</identifier>
        <datestamp>2008-04-16</datestamp>
      </header>
      <metadata>
        <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
          <id>0804.2273</id>
          <created>2008-04-14</created>
          <title>Example paper</title>
          <categories>cs.DL cs.NI</categories>
          <abstract>Example abstract.</abstract>
        </arXiv>
      </metadata>
    </record>
    <resumptionToken>next-token</resumptionToken>
  </ListRecords>
</OAI-PMH>
"""


def make_response(content: bytes = XML_RESPONSE) -> httpx.Response:
    """Build a successful mock response."""
    return httpx.Response(200, content=content, request=httpx.Request("GET", "https://example.test"))


def test_initial_request_uses_metadata_prefix() -> None:
    with ArxivOAIClient(min_request_interval=0) as client, patch.object(client._http, "get", return_value=make_response()) as get:
        page = client.fetch_page()

    assert page.next_token == "next-token"
    assert page.records[0].paper is not None
    assert page.records[0].paper.arxiv_id == "0804.2273"
    assert get.call_args.kwargs["params"] == {
        "verb": "ListRecords",
        "metadataPrefix": "arXiv",
    }


def test_resumption_request_uses_token_only() -> None:
    with ArxivOAIClient(min_request_interval=0) as client, patch.object(client._http, "get", return_value=make_response()) as get:
        client.fetch_page("next-token")

    assert get.call_args.kwargs["params"] == {
        "verb": "ListRecords",
        "resumptionToken": "next-token",
    }


def test_http_errors_propagate() -> None:
    request = httpx.Request("GET", "https://example.test")
    response = httpx.Response(503, request=request)

    with ArxivOAIClient(min_request_interval=0) as client, patch.object(client._http, "get", return_value=response), pytest.raises(httpx.HTTPStatusError):
            client.fetch_page()