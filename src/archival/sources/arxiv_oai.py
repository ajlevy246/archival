
"""Parse ArXiv OAI-PMH XML responses into source-neutral paper records."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Literal

from archival.types import PaperRecord


@dataclass(frozen=True)
class OAIRecord:
    """A paper record or deletion event received through OAI-PMH."""

    status: Literal["active", "deleted"]
    datestamp: str
    paper: PaperRecord | None
    identifier: str

class OAIParseError(ValueError):
    """Raised when an OAI-PMH response cannot be parsed safely."""


@dataclass(frozen=True)
class ParsedHarvestPage:
    """Parsed records and pagination state from one ListRecords response."""

    records: tuple[OAIRecord, ...]
    next_token: str | None
    completed: bool


def _local_name(tag: str) -> str:
    """Return an XML tag's local name, ignoring its namespace."""
    return tag.rsplit("}", 1)[-1]


def _child(element: ET.Element, name: str) -> ET.Element | None:
    """Find a direct child by local tag name."""
    return next(
        (child for child in element if _local_name(child.tag) == name),
        None,
    )


def _text(element: ET.Element | None) -> str | None:
    """Extract text, collapse whitespace, and return None when empty."""
    if element is None:
        return None
    value = " ".join("".join(element.itertext()).split())
    return value or None


def _required_text(element: ET.Element, name: str) -> str:
    """Extract a required direct-child text field."""
    value = _text(_child(element, name))
    if value is None:
        raise OAIParseError(f"Missing or empty required XML field: {name}")
    return value


def _arxiv_id(identifier: str) -> str:
    """Normalize a canonical OAI identifier to its arXiv ID."""
    prefix = "oai:arXiv.org:"
    if identifier.startswith(prefix):
        return identifier[len(prefix):]
    return identifier


def _parse_record(element: ET.Element) -> OAIRecord:
    """Parse one OAI-PMH record element."""
    header = _child(element, "header")
    if header is None:
        raise OAIParseError("Record is missing its OAI header")

    identifier = _required_text(header, "identifier")
    datestamp = _required_text(header, "datestamp")
    status = header.attrib.get("status")

    if status == "deleted":
        return OAIRecord(
            status="deleted",
            datestamp=datestamp,
            paper=None,
            identifier=identifier,
        )

    metadata = _child(element, "metadata")
    if metadata is None or len(metadata) == 0:
        raise OAIParseError(
            f"Active record {identifier!r} is missing metadata"
        )

    # The arXiv metadata document is nested under OAI-PMH's metadata element.
    arxiv_metadata = metadata[0]

    paper_id = _text(_child(arxiv_metadata, "id"))
    if paper_id is None:
        paper_id = _arxiv_id(identifier)

    title = _required_text(arxiv_metadata, "title")
    abstract = _required_text(arxiv_metadata, "abstract")
    created = _text(_child(arxiv_metadata, "created"))
    categories_text = _text(_child(arxiv_metadata, "categories"))
    comments = _text(_child(arxiv_metadata, "comments"))
    submitter = _text(_child(arxiv_metadata, "submitter"))

    categories = tuple(categories_text.split()) if categories_text else ()

    paper = PaperRecord(
        arxiv_id=paper_id,
        title=title,
        abstract=abstract,
        categories=categories,
        date=created,
        comments=comments,
        submitter=submitter,
    )

    return OAIRecord(
        status="active",
        datestamp=datestamp,
        paper=paper,
        identifier=identifier,
    )


def parse_list_records(xml_data: str | bytes) -> ParsedHarvestPage:
    """Parse an OAI-PMH ListRecords XML response."""
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as exc:
        raise OAIParseError("Response is not valid XML") from exc

    if _local_name(root.tag) != "OAI-PMH":
        raise OAIParseError("Unexpected XML root; expected OAI-PMH")

    error = _child(root, "error")
    if error is not None:
        code = error.attrib.get("code", "unknown")
        message = _text(error) or "No error description provided"
        raise OAIParseError(f"OAI-PMH error {code!r}: {message}")

    response = _child(root, "ListRecords")
    if response is None:
        raise OAIParseError("Response does not contain ListRecords")

    records = tuple(
        _parse_record(element)
        for element in response
        if _local_name(element.tag) == "record"
    )

    token_element = _child(response, "resumptionToken")
    next_token = _text(token_element)

    return ParsedHarvestPage(
        records=records,
        next_token=next_token,
        completed=next_token is None,
    )