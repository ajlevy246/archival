from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PaperRecord:
    arxiv_id: str
    title: str
    abstract: str
    categories: tuple[str, ...] = ()
    date: str | None = None
    comments: str | None = None
    submitter: str | None = None


@dataclass(frozen=True)
class Representation:
    text: str
    fields: dict[str, Any]


@dataclass(frozen=True)
class PaperMeta:
    arxiv_id: str
    title: str
    categories: tuple[str, ...] = ()
    date: str | None = None


@dataclass(frozen=True)
class Hit:
    row: int
    score: float