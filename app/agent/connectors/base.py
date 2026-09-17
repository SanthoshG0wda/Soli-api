"""Base connector ABC — all connectors implement this interface."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class BaseConnector(ABC):
    """
    A Connector wraps a specific data source (IndiaKanoon, India Code, etc.)
    and exposes one or more async methods that tools call.
    Each connector is responsible for HTTP, parsing, and returning clean dicts.
    """

    @property
    @abstractmethod
    def source_name(self) -> str: ...

    @property
    @abstractmethod
    def source_domain(self) -> str: ...

    @property
    @abstractmethod
    def doc_type(self) -> str: ...

    @property
    def authority_score(self) -> float:
        return 0.7

    def _base_result(self, title: str, url: str, snippet: str = "") -> dict:
        return {
            "title": title,
            "url": url,
            "snippet": snippet[:400],
            "source_name": self.source_name,
            "source_domain": self.source_domain,
            "doc_type": self.doc_type,
            "authority_score": self.authority_score,
            "is_pdf": url.lower().endswith(".pdf"),
        }
