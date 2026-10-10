
"""Tests for parsing ArXiv OAI-PMH responses."""

import pytest

from archival.sources.arxiv_oai import (
    OAIParseError,
    parse_list_records,
)

OAI_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <responseDate>2026-10-10T04:00:00Z</responseDate>
  <request verb="ListRecords" metadataPrefix="arXiv">
    https://oaipmh.arxiv.org/oai
  </request>
  <ListRecords>
    <record>
      <header>
        <identifier>oai:arXiv.org:2601.12345</identifier>
        <datestamp>2026-10-08</datestamp>
        <setSpec>cs</setSpec>
      </header>
      <metadata>
        <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
          <id>2601.12345</id>
          <created>2026-01-15</created>
          <title>A Test Paper</title>
          <authors>
            <author><keyname>Example</keyname></author>
          </authors>
          <categories>cs.AI cs.LG</categories>
          <abstract>An example abstract.</abstract>
          <comments>10 pages</comments>
          <submitter>Example Submitter</submitter>
        </arXiv>
      </metadata>
    </record>
    <record>
      <header status="deleted">
        <identifier>oai:arXiv.org:2601.54321</identifier>
        <datestamp>2026-10-09</datestamp>
      </header>
    </record>
    <resumptionToken>next-page-token</resumptionToken>
  </ListRecords>
</OAI-PMH>
"""


def test_parse_active_and_deleted_records() -> None:
    page = parse_list_records(OAI_XML)

    assert len(page.records) == 2
    assert page.completed is False
    assert page.next_token == "next-page-token"

    active, deleted = page.records

    assert active.status == "active"
    assert active.identifier == "oai:arXiv.org:2601.12345"
    assert active.datestamp == "2026-10-08"
    assert active.paper is not None
    assert active.paper.arxiv_id == "2601.12345"
    assert active.paper.title == "A Test Paper"
    assert active.paper.abstract == "An example abstract."
    assert active.paper.categories == ("cs.AI", "cs.LG")
    assert active.paper.date == "2026-01-15"
    assert active.paper.comments == "10 pages"
    assert active.paper.submitter == "Example Submitter"

    assert deleted.status == "deleted"
    assert deleted.paper is None
    assert deleted.datestamp == "2026-10-09"


def test_final_page_has_no_token() -> None:
    xml = OAI_XML.replace(
        "<resumptionToken>next-page-token</resumptionToken>",
        "<resumptionToken/>",
    )

    page = parse_list_records(xml)

    assert page.next_token is None
    assert page.completed is True


def test_oai_protocol_error_is_reported() -> None:
    xml = """\
    <OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
      <error code="badResumptionToken">Token expired</error>
    </OAI-PMH>
    """

    with pytest.raises(OAIParseError, match="badResumptionToken"):
        parse_list_records(xml)


def test_malformed_xml_is_rejected() -> None:
    with pytest.raises(OAIParseError, match="valid XML"):
        parse_list_records("<OAI-PMH>")