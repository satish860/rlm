"""
RLM Schemas - Re-export of built-in extraction schemas.

This module provides convenient access to built-in schemas via rlm.schemas.

Example:
    import rlm

    result = rlm.extract("contacts.pdf", schema=list[rlm.schemas.Contact])
"""

from rlm.extraction.schemas import (
    Contact,
    Invoice,
    Entity,
    TableRow,
    KeyValuePair,
    FinancialFigure,
)

__all__ = [
    "Contact",
    "Invoice",
    "Entity",
    "TableRow",
    "KeyValuePair",
    "FinancialFigure",
]
