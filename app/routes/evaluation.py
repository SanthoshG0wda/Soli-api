"""
Evaluation & Trust Benchmark API Routes
=======================================
Exposes endpoints to run live evaluation benchmarks and inspect trust metrics.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.evaluations.runner import run_legal_benchmark

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluation", tags=["evaluation"])

# In-memory and disk cache for the latest evaluation report
_latest_report: dict[str, Any] | None = None
_CACHE_FILE = Path(settings.STORAGE_BASE_PATH) / "evaluation_report.json"


def _save_report(report: dict[str, Any]) -> None:
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    except Exception as exc:
        logger.warning("Failed to save evaluation report to cache file: %s", exc)


def _load_cached_report() -> dict[str, Any] | None:
    try:
        if _CACHE_FILE.exists():
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load evaluation report from cache file: %s", exc)
    return None


@router.post("/run", status_code=status.HTTP_200_OK)
def run_evaluation_endpoint(db: Session = Depends(get_db)):
    """
    Executes the multi-pillar legal benchmark test suite against the RAG
    vector database and returns the full evaluation audit report.
    """
    global _latest_report
    report = run_legal_benchmark(db=db)
    _latest_report = report
    _save_report(report)
    return report


@router.get("/latest", status_code=status.HTTP_200_OK)
def get_latest_evaluation(db: Session = Depends(get_db)):
    """
    Returns the most recent evaluation report from memory or disk cache immediately.
    Executes a fresh benchmark only if no report has ever been generated.
    """
    global _latest_report
    if _latest_report is not None:
        return _latest_report

    cached = _load_cached_report()
    if cached is not None:
        _latest_report = cached
        return _latest_report

    _latest_report = run_legal_benchmark(db=db)
    _save_report(_latest_report)
    return _latest_report

