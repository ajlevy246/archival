from archival.types import Hit, PaperMeta, PaperRecord, Representation


def test_paper_record():
    paper = PaperRecord(
        arxiv_id="2601.12345",
        title="A Test Paper",
        abstract="This is a test abstract.",
        categories=("cs.AI", "cs.LG"),
        date="2026-01-15",
    )

    assert paper.arxiv_id == "2601.12345"
    assert paper.title == "A Test Paper"
    assert paper.categories == ("cs.AI", "cs.LG")


def test_representation():
    representation = Representation(
        text="A Test Paper\n\nThis is a test abstract.",
        fields={
            "title": "A Test Paper",
            "abstract": "This is a test abstract.",
        },
    )

    assert representation.text.startswith("A Test Paper")
    assert representation.fields["title"] == "A Test Paper"


def test_paper_meta():
    meta = PaperMeta(
        arxiv_id="2601.12345",
        title="A Test Paper",
        categories=("cs.AI",),
    )

    assert meta.arxiv_id == "2601.12345"
    assert meta.title == "A Test Paper"


def test_hit():
    hit = Hit(row=17, score=0.95)

    assert hit.row == 17
    assert hit.score == 0.95