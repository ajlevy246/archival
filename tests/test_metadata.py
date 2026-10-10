"""Tests for metadata persistence and harvest checkpoints."""

from archival.metadata import MetadataStore
from archival.types import PaperRecord


def make_paper(
    arxiv_id: str = "2601.12345",
    title: str = "A Test Paper",
) -> PaperRecord:
    """Create a paper record for metadata tests."""
    return PaperRecord(
        arxiv_id=arxiv_id,
        title=title,
        abstract="An example abstract.",
        categories=("cs.AI", "cs.LG"),
        date="2026-01-15",
    )


def test_upsert_and_retrieve_paper(tmp_path):
    store = MetadataStore(tmp_path / "metadata.sqlite")
    paper = make_paper()

    store.upsert_papers([paper])

    assert store.get_paper(paper.arxiv_id) == paper
    assert store.count_papers() == 1


def test_upsert_updates_existing_paper(tmp_path):
    store = MetadataStore(tmp_path / "metadata.sqlite")

    store.upsert_papers([make_paper()])
    store.upsert_papers([make_paper(title="Updated Title")])

    assert store.count_papers() == 1
    assert (record := store.get_paper("2601.12345")) is not None and record.title == "Updated Title"


def test_harvest_page_and_checkpoint_are_persisted(tmp_path):
    path = tmp_path / "metadata.sqlite"
    store = MetadataStore(path)

    store.commit_harvest_page(
        [make_paper()],
        next_token="next-page-token",
    )

    assert store.count_papers() == 1
    assert store.get_resumption_token() == "next-page-token"
    assert not store.is_harvest_complete()


def test_completed_harvest_has_no_next_token(tmp_path):
    store = MetadataStore(tmp_path / "metadata.sqlite")

    store.commit_harvest_page(
        [make_paper()],
        next_token=None,
        completed=True,
    )

    assert store.get_resumption_token() is None
    assert store.is_harvest_complete()


def test_checkpoint_survives_reopening_database(tmp_path):
    path = tmp_path / "metadata.sqlite"

    MetadataStore(path).commit_harvest_page(
        [make_paper()],
        next_token="resume-here",
    )

    reopened = MetadataStore(path)

    assert reopened.get_resumption_token() == "resume-here"
    assert reopened.get_paper("2601.12345") == make_paper()