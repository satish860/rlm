"""
RLM Built-in Schemas - Common extraction schemas ready to use.

These schemas can be used directly or as templates for custom schemas.
All schemas include a 'page' field for citation tracking.

Example:
    import rlm

    # Use built-in schema
    result = rlm.extract("contacts.pdf", schema=list[rlm.schemas.Contact])

    # Or define custom schema
    from pydantic import BaseModel

    class CustomEntity(BaseModel):
        name: str
        description: str
        page: int  # Required for citation tracking

    result = rlm.extract("doc.pdf", schema=list[CustomEntity])
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class Contact(BaseModel):
    """
    Contact information extracted from a document.

    Use for extracting people, companies, or organizations with
    their contact details.
    """
    name: str = Field(..., description="Full name of person or organization")
    company: Optional[str] = Field(default=None, description="Company or organization name")
    title: Optional[str] = Field(default=None, description="Job title or role")
    email: Optional[str] = Field(default=None, description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    address: Optional[str] = Field(default=None, description="Physical address")
    page: int = Field(..., description="Page number where found")


class Invoice(BaseModel):
    """
    Invoice data extracted from a document.

    Use for extracting billing and payment information.
    """
    invoice_number: Optional[str] = Field(default=None, description="Invoice ID/number")
    vendor: str = Field(..., description="Vendor or seller name")
    customer: Optional[str] = Field(default=None, description="Customer or buyer name")
    date: Optional[str] = Field(default=None, description="Invoice date")
    due_date: Optional[str] = Field(default=None, description="Payment due date")
    subtotal: Optional[float] = Field(default=None, description="Subtotal before tax")
    tax: Optional[float] = Field(default=None, description="Tax amount")
    total_amount: float = Field(..., description="Total amount due")
    currency: str = Field(default="USD", description="Currency code")
    line_items: List[dict] = Field(default_factory=list, description="Individual line items")
    page: int = Field(..., description="Page number where found")


class Entity(BaseModel):
    """
    Named entity extracted from a document.

    Use for general entity extraction (people, places, organizations,
    products, etc.)
    """
    name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Type: person, org, place, product, etc.")
    description: Optional[str] = Field(default=None, description="Brief description")
    attributes: dict = Field(default_factory=dict, description="Additional attributes")
    page: int = Field(..., description="Page number where found")


class TableRow(BaseModel):
    """
    A row extracted from a table in a document.

    Use for extracting tabular data. The 'columns' dict maps
    column names to values.
    """
    row_number: int = Field(..., description="Row number in table (1-indexed)")
    columns: dict = Field(..., description="Column name -> value mapping")
    table_name: Optional[str] = Field(default=None, description="Name/title of the table")
    page: int = Field(..., description="Page number where found")


class KeyValuePair(BaseModel):
    """
    A key-value pair extracted from a document.

    Use for extracting form fields, metadata, or structured
    key-value data.
    """
    key: str = Field(..., description="The key/label/field name")
    value: str = Field(..., description="The value")
    section: Optional[str] = Field(default=None, description="Section where found")
    page: int = Field(..., description="Page number where found")


class FinancialFigure(BaseModel):
    """
    Financial figure extracted from a document.

    Use for extracting monetary values with context.
    """
    label: str = Field(..., description="Label/description of the figure")
    value: float = Field(..., description="Numeric value")
    currency: str = Field(default="USD", description="Currency code")
    period: Optional[str] = Field(default=None, description="Time period (Q1 2024, FY2023, etc.)")
    category: Optional[str] = Field(default=None, description="Category: revenue, expense, profit, etc.")
    page: int = Field(..., description="Page number where found")
