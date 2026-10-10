"""SQLite persistence for harvested paper metadata and ingestion progress."""

import json
import sqlite3
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path

from archival.types import PaperRecord


class MetadataStore:
    """Persist paper metadata and OAI-PMH harvesting checkpoints."""

    def __init__(self, path: Path) -> None:
        """Open or create a metadata database at the given path."""
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection]:
        """Provide a connection and ensure it is closed after use."""
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row

        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        """Create the database schema if it does not already exist."""
        with self._connection() as connection, connection:
            connection.execute(
                """
                    CREATE TABLE IF NOT EXISTS papers (
                        arxiv_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        abstract TEXT NOT NULL,
                        categories TEXT NOT NULL,
                        date TEXT,
                        comments TEXT,
                        submitter TEXT
                    )
                    """
            )
            connection.execute(
                """
                    CREATE TABLE IF NOT EXISTS harvest_state (
                        source TEXT PRIMARY KEY,
                        resumption_token TEXT,
                        completed INTEGER NOT NULL DEFAULT 0
                    )
                    """
            )

    def upsert_papers(self, papers: Sequence[PaperRecord]) -> None:
        """Insert or update papers, keyed by their arXiv IDs."""
        rows = [
            (
                paper.arxiv_id,
                paper.title,
                paper.abstract,
                json.dumps(paper.categories),
                paper.date,
                paper.comments,
                paper.submitter,
            )
            for paper in papers
        ]

        with self._connection() as connection, connection:
            connection.executemany(
                """
                    INSERT INTO papers (
                        arxiv_id, title, abstract, categories,
                        date, comments, submitter
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(arxiv_id) DO UPDATE SET
                        title = excluded.title,
                        abstract = excluded.abstract,
                        categories = excluded.categories,
                        date = excluded.date,
                        comments = excluded.comments,
                        submitter = excluded.submitter
                    """,
                rows,
            )

    def commit_harvest_page(
        self,
        papers: Sequence[PaperRecord],
        next_token: str | None,
        *,
        completed: bool = False,
        source: str = "arxiv-oai",
    ) -> None:
        """Atomically store a harvested page and its next checkpoint."""
        rows = [
            (
                paper.arxiv_id,
                paper.title,
                paper.abstract,
                json.dumps(paper.categories),
                paper.date,
                paper.comments,
                paper.submitter,
            )
            for paper in papers
        ]

        with self._connection() as connection, connection:
            connection.executemany(
                """
                    INSERT INTO papers (
                        arxiv_id, title, abstract, categories,
                        date, comments, submitter
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(arxiv_id) DO UPDATE SET
                        title = excluded.title,
                        abstract = excluded.abstract,
                        categories = excluded.categories,
                        date = excluded.date,
                        comments = excluded.comments,
                        submitter = excluded.submitter
                    """,
                rows,
            )
            connection.execute(
                """
                    INSERT INTO harvest_state (
                        source, resumption_token, completed
                    )
                    VALUES (?, ?, ?)
                    ON CONFLICT(source) DO UPDATE SET
                        resumption_token = excluded.resumption_token,
                        completed = excluded.completed
                    """,
                (source, next_token, int(completed)),
            )

    def get_resumption_token(self, source: str = "arxiv-oai") -> str | None:
        """Return the saved OAI-PMH token, or None if harvesting has not started."""
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT resumption_token
                FROM harvest_state
                WHERE source = ?
                """,
                (source,),
            ).fetchone()

        return None if row is None else row["resumption_token"]

    def is_harvest_complete(self, source: str = "arxiv-oai") -> bool:
        """Return whether the saved harvest has completed."""
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT completed
                FROM harvest_state
                WHERE source = ?
                """,
                (source,),
            ).fetchone()

        return row is not None and bool(row["completed"])

    def get_paper(self, arxiv_id: str) -> PaperRecord | None:
        """Retrieve one paper by arXiv ID."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM papers WHERE arxiv_id = ?",
                (arxiv_id,),
            ).fetchone()

        if row is None:
            return None

        return PaperRecord(
            arxiv_id=row["arxiv_id"],
            title=row["title"],
            abstract=row["abstract"],
            categories=tuple(json.loads(row["categories"])),
            date=row["date"],
            comments=row["comments"],
            submitter=row["submitter"],
        )

    def count_papers(self) -> int:
        """Return the number of unique papers in the database."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM papers"
            ).fetchone()

        return int(row["count"])