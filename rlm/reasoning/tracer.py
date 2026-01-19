"""
RLM Reasoning Tracer - Track thinking, citations, and progress.

The ReasoningTracer provides transparency into the extraction process by:
- Recording structured thinking (think())
- Tracking evidence citations (cite())
- Evaluating extraction progress (evaluate_progress())

Example:
    tracer = ReasoningTracer(total_pages=50)

    tracer.think("Document has 5 sections. Will extract contacts.")
    tracer.cite("John Smith, CEO", page=3, note="First contact found")

    confidence = tracer.evaluate_progress(
        records_extracted=10,
        pages_covered=[1, 2, 3, 4, 5]
    )
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class ThinkingEntry:
    """A recorded thought from the extraction process."""
    timestamp: str
    thought: str


@dataclass
class CitationEntry:
    """An evidence citation from the document."""
    snippet: str
    page: int
    note: str = ""


@dataclass
class ConfidenceEntry:
    """A progress evaluation snapshot."""
    records: int
    pages_covered: int
    total_pages: int
    coverage: float
    issues: str
    notes: str
    confidence: float


class ReasoningTracer:
    """
    Tracks reasoning, citations, and progress during extraction.

    Provides methods for the root model to:
    - Structure thinking before actions
    - Record evidence with citations
    - Self-assess extraction progress

    All data is stored for inclusion in ExtractionResult.
    """

    def __init__(
        self,
        total_pages: int = 1,
        progress_callback: Callable[[str], None] = None
    ):
        """
        Initialize reasoning tracer.

        Args:
            total_pages: Total pages in document (for coverage calculation)
            progress_callback: Function called with progress messages
        """
        self.total_pages = total_pages
        self._progress_callback = progress_callback

        # State
        self.thinking_log: List[Dict[str, Any]] = []
        self.citations: List[Dict[str, Any]] = []
        self.confidence_history: List[Dict[str, Any]] = []

    def _progress(self, msg: str):
        """Send progress message."""
        if self._progress_callback:
            self._progress_callback(msg)

    def think(self, reasoning: str) -> str:
        """
        Structure reasoning before taking action.

        Use to break down problems, plan approach, note observations.
        All thoughts are logged for transparency.

        Args:
            reasoning: The reasoning/thought to record

        Returns:
            Confirmation message

        Example:
            think("Document has 5 sections. Section 1 (pages 1-10) contains contacts.")
            think("Data format: S.No, Name, Address, Phone. Will use structured extraction.")
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "thought": reasoning
        }
        self.thinking_log.append(entry)
        self._progress(f"THINK: {reasoning[:80]}...")
        return f"Thought recorded. Total thoughts: {len(self.thinking_log)}"

    def cite(self, snippet: str, page: int, note: str = "") -> str:
        """
        Record evidence citation from the document.

        Args:
            snippet: Exact verbatim text from source
            page: Page number where found
            note: Your interpretation or why this matters

        Returns:
            Confirmation message

        Example:
            cite("John Smith, CEO, Acme Corp", page=3, note="First contact entry")
            cite("Total Revenue: $1.2M", page=15, note="Q3 financial figure")
        """
        entry = {
            "snippet": snippet,
            "page": page,
            "note": note
        }
        self.citations.append(entry)
        return f"Citation recorded. Total citations: {len(self.citations)}"

    def evaluate_progress(
        self,
        records_extracted: int = None,
        pages_covered: List[int] = None,
        issues: str = "",
        notes: str = "",
        records_list: List = None
    ) -> float:
        """
        Self-assess extraction progress.

        Call periodically to decide if more work is needed.
        Returns confidence score 0.0 to 1.0.

        Args:
            records_extracted: How many records found so far
            pages_covered: Which pages have been processed
            issues: Any problems encountered
            notes: Additional observations
            records_list: Optional list to count records from

        Returns:
            Confidence score (target >= 0.95 before final_answer)

        Example:
            confidence = evaluate_progress(
                records_extracted=50,
                pages_covered=[1,2,3,4,5,6,7,8,9,10],
                notes="Sections 1-2 complete"
            )
            if confidence < 0.95:
                # Continue extraction
                pass
        """
        # Count records
        if records_extracted is not None:
            records_count = records_extracted
        elif records_list is not None:
            records_count = len(records_list)
        else:
            records_count = 0

        pages_done = pages_covered or []

        # Calculate confidence based on coverage
        coverage = len(pages_done) / self.total_pages if self.total_pages > 0 else 0
        has_records = 1.0 if records_count > 0 else 0.0
        has_issues = 0.8 if issues else 1.0

        confidence = (coverage * 0.5 + has_records * 0.3 + has_issues * 0.2)
        confidence = min(1.0, max(0.0, confidence))

        entry = {
            "records": records_count,
            "pages_covered": len(pages_done),
            "total_pages": self.total_pages,
            "coverage": coverage,
            "issues": issues,
            "notes": notes,
            "confidence": confidence
        }
        self.confidence_history.append(entry)

        self._progress(f"EVALUATE: confidence={confidence:.2f}, records={records_count}, coverage={coverage:.1%}")
        return confidence

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of reasoning trace.

        Returns:
            Dict with thinking_log, citations, confidence_history
        """
        return {
            "thinking_log": self.thinking_log,
            "citations": self.citations,
            "confidence_history": self.confidence_history,
            "final_confidence": self.confidence_history[-1]["confidence"] if self.confidence_history else 0.0
        }

    def reset(self):
        """Clear all reasoning state."""
        self.thinking_log.clear()
        self.citations.clear()
        self.confidence_history.clear()

    def get_functions(self) -> Dict[str, Callable]:
        """
        Get functions to inject into REPL namespace.

        Returns:
            Dict mapping function names to callables
        """
        return {
            "think": self.think,
            "cite": self.cite,
            "evaluate_progress": self.evaluate_progress,
        }
