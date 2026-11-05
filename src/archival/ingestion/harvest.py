"""Harvester code for ArXiv's OAI-PMH API w/ Sickle"""
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from time import time

import sickle
from sickle import iterator, models


@dataclass(frozen=True)
class ArticleMetadata:
    archival_id: str
    title: str
    abstract: str
    authors: tuple[str]
    date: str
    raw: dict[str,str] # raw jsonl encoding of the metadata

    def as_json(self):
        return json.dumps(self.__dict__, separators=(',', ':'))
        

class Harvester:
    ARXIV_OAI_BASEURL = "https://oaipmh.arxiv.org/oai"
    METADATA_PREFIX   = "oai_dc"
    DATA_DIR = Path.home() / ".archival" / "raw"

    @staticmethod
    def _id_to_filename(archival_id: str) -> str:
        """Convert an ArXiv id to a local filename (.jsonl filetype)"""
        return archival_id.removeprefix("oai:arXiv.org:").replace("/", "_") + ".jsonl"

    def __init__(self):
        self.log = logging.getLogger(__name__)

        self.harvester = sickle.Sickle(self.ARXIV_OAI_BASEURL)

        # initial fetch of sets
        self._sets: dict[str,str] = {
            s.setName: s.setSpec
            for s in self.harvester.ListSets()
        }

    @property
    def sets(self) -> list[str]:
        """List of paper subjects available - e.g. Physics, Math, etc."""
        return list(self._sets.keys())

    def save_records(self, set_name: str) -> None:
        """Save a set of raw record data to disk."""
        if set_name not in self._sets:
            raise ValueError(f"Unrecognized set: {set_name}")

        save_dir = self.DATA_DIR / set_name
        save_dir.mkdir(parents=True, exist_ok=True)

        self.log.debug(f"Harvesting records in the '{set_name}' category")
        self.log.debug(f"- saving locally to '{save_dir}'")
        harvest_start = time()

        for metadata in self._list_records(set_name):
            file_name = self._id_to_filename(metadata.archival_id)

            with (save_dir/file_name).open('w') as file:
                file.write(metadata.as_json())

        harvest_end = time()
        self.log.debug(f"Done in {(harvest_end - harvest_start)/1000:.2f} seconds")

    def _list_records(self, set_name: str) -> Iterator[ArticleMetadata]:
        """Iterate over records for a given set."""
        if set_name not in self._sets:
            raise ValueError(f"Unrecognized set: {set_name}")

        record_iterator: iterator.OAIItemIterator = self.harvester.ListRecords(
            ignore_deleted=True,
            metadataPrefix=self.METADATA_PREFIX,
            set=self._sets[set_name],
        )

        for record in record_iterator:
            if (normalized := self._parse_record(record)) is not None:
                yield normalized
            else:
                self.log.warning("skipping record...")

    def _parse_record(self, record: models.Record) -> ArticleMetadata | None:
        """Normalize an Arxiv OAI-PMH ListRecords response."""

        # paper's ArXiv ID better be present
        if not (identifier := record.header.identifier):
            self.log.warning("  - missing ArXiv identifier")
            return None

        # verify that the data we require is actually present in the record
        data = record.metadata
        if 'title' not in data or not (title := data['title']):
            self.log.warning("  - missing title")
            return None

        if 'creator' not in data or not (authors := data['creator']):
            self.log.warning("  - missing authors list")
            return None
            
        if 'description' not in data or not (abstract := data['description']):
            self.log.warning("  - missing paper abstract")
            return None

        if 'date' not in data or not (date := data['date']):
            self.log.warning("  - missing paper publication date")
            return None

        return ArticleMetadata(
            archival_id=identifier,
            title=title,
            abstract=abstract,
            authors=authors,
            date=date,
            raw=data
        )