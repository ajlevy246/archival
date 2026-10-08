"""Paper representation strategies used to construct searchable text."""

from archival.types import PaperRecord, Representation


class TitleAbstractRepresenter:
    """Represent a paper using its title and abstract."""

    def represent(self, paper: PaperRecord) -> Representation:
        """Create a representation containing the paper's title and abstract."""
        text = f"{paper.title}\n\n{paper.abstract}"

        return Representation(
            text=text,
            fields={
                "title": paper.title,
                "abstract": paper.abstract,
            },
        )