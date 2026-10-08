"""Tests for paper representation strategies."""

from archival.representers import TitleAbstractRepresenter
from archival.types import PaperRecord


def test_title_abstract_representer():
    paper = PaperRecord(
        arxiv_id="2601.12345",
        title="A Test Paper",
        abstract="This is a test abstract.",
    )

    representation = TitleAbstractRepresenter().represent(paper)

    assert representation.text == "A Test Paper\n\nThis is a test abstract."
    assert representation.fields == {
        "title": "A Test Paper",
        "abstract": "This is a test abstract.",
    }