"""
RLM Session Manager - Save and restore extraction sessions.

Sessions allow:
- Pausing and resuming long extractions
- Recovering from errors
- Reviewing extraction history

Example:
    manager = SessionManager(sessions_dir="sessions")

    # Save current state
    manager.save_session("contacts_extraction", {
        "records": records,
        "citations": citations,
        "thinking_log": thinking_log
    })

    # Later, restore
    state = manager.load_session("contacts_extraction")
    records = state["records"]
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


class SessionManager:
    """
    Manages extraction session persistence.

    Sessions store:
    - Extracted records
    - Citations
    - Thinking log
    - Confidence history
    - Metadata (document, schema, timestamp)
    """

    def __init__(self, sessions_dir: str = "sessions"):
        """
        Initialize session manager.

        Args:
            sessions_dir: Directory to store session files
        """
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def save_session(
        self,
        name: str,
        state: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Save extraction session state.

        Args:
            name: Session name (used as filename)
            state: Dict with records, citations, thinking_log, etc.
            metadata: Optional metadata (document_path, schema, etc.)

        Returns:
            Path to saved session file

        Example:
            manager.save_session("my_extraction", {
                "records": records,
                "citations": citations,
                "thinking_log": thinking_log,
                "confidence_history": confidence_history,
                "extracted_data": extracted_data
            }, metadata={
                "document_path": "contacts.pdf",
                "schema": "Contact"
            })
        """
        session_data = {
            "name": name,
            "saved_at": datetime.now().isoformat(),
            "state": state,
            "metadata": metadata or {}
        }

        filepath = self.sessions_dir / f"{name}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False, default=str)

        return str(filepath)

    def load_session(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Load a previously saved session.

        Args:
            name: Session name

        Returns:
            Session data dict, or None if not found

        Example:
            session = manager.load_session("my_extraction")
            if session:
                records = session["state"]["records"]
                citations = session["state"]["citations"]
        """
        filepath = self.sessions_dir / f"{name}.json"
        if not filepath.exists():
            return None

        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all saved sessions.

        Returns:
            List of session info dicts with name, saved_at, metadata
        """
        sessions = []
        for filepath in self.sessions_dir.glob("*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    sessions.append({
                        "name": data.get("name", filepath.stem),
                        "saved_at": data.get("saved_at"),
                        "metadata": data.get("metadata", {}),
                        "records_count": len(data.get("state", {}).get("records", [])),
                        "filepath": str(filepath)
                    })
            except (json.JSONDecodeError, KeyError):
                # Skip invalid session files
                pass

        # Sort by saved_at descending (most recent first)
        sessions.sort(key=lambda x: x.get("saved_at", ""), reverse=True)
        return sessions

    def delete_session(self, name: str) -> bool:
        """
        Delete a saved session.

        Args:
            name: Session name

        Returns:
            True if deleted, False if not found
        """
        filepath = self.sessions_dir / f"{name}.json"
        if filepath.exists():
            filepath.unlink()
            return True
        return False

    def session_exists(self, name: str) -> bool:
        """Check if a session exists."""
        return (self.sessions_dir / f"{name}.json").exists()

    def format_sessions_table(self) -> str:
        """
        Format sessions list as a text table for CLI display.

        Returns:
            Formatted table string
        """
        sessions = self.list_sessions()
        if not sessions:
            return "No saved sessions found."

        lines = [
            "Saved Sessions:",
            "-" * 70,
            f"{'Name':<25} {'Records':<10} {'Saved At':<25} {'Document'}",
            "-" * 70,
        ]

        for s in sessions:
            name = s["name"][:24]
            records = str(s.get("records_count", 0))
            saved_at = s.get("saved_at", "")[:24]
            doc = s.get("metadata", {}).get("document_path", "")[:20]
            lines.append(f"{name:<25} {records:<10} {saved_at:<25} {doc}")

        lines.append("-" * 70)
        lines.append(f"Total: {len(sessions)} session(s)")
        return "\n".join(lines)

    def get_session_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed info about a session.

        Args:
            name: Session name

        Returns:
            Session info dict or None if not found
        """
        session = self.load_session(name)
        if not session:
            return None

        state = session.get("state", {})
        return {
            "name": session.get("name", name),
            "saved_at": session.get("saved_at"),
            "metadata": session.get("metadata", {}),
            "records_count": len(state.get("records", [])),
            "citations_count": len(state.get("citations", [])),
            "thoughts_count": len(state.get("thinking_log", [])),
            "total_pages": state.get("total_pages", 0),
            "final_confidence": (
                state.get("confidence_history", [{}])[-1].get("confidence", 0)
                if state.get("confidence_history") else 0
            )
        }

    def get_functions(self, namespace: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get functions to inject into REPL namespace.

        Args:
            namespace: Optional namespace dict to read state from

        Returns:
            Dict mapping function names to callables
        """
        def save_session_fn(name: str) -> str:
            """Save complete session state for later resumption."""
            state = {
                "records": namespace.get("records", []) if namespace else [],
                "citations": namespace.get("citations", []) if namespace else [],
                "thinking_log": namespace.get("thinking_log", []) if namespace else [],
                "confidence_history": namespace.get("confidence_history", []) if namespace else [],
                "extracted_data": namespace.get("extracted_data", {}) if namespace else {},
                "total_pages": namespace.get("total_pages", 0) if namespace else 0,
            }
            filepath = self.save_session(name, state)
            return f"Session saved to {filepath}"

        def load_session_fn(name: str) -> str:
            """Load a previously saved session state."""
            session = self.load_session(name)
            if not session:
                return f"Session not found: {name}"

            state = session.get("state", {})
            if namespace is not None:
                namespace["records"] = state.get("records", [])
                namespace["citations"] = state.get("citations", [])
                namespace["thinking_log"] = state.get("thinking_log", [])
                namespace["confidence_history"] = state.get("confidence_history", [])
                namespace["extracted_data"] = state.get("extracted_data", {})

            records_count = len(state.get("records", []))
            citations_count = len(state.get("citations", []))
            return f"Session loaded: {records_count} records, {citations_count} citations"

        return {
            "save_session": save_session_fn,
            "load_session": load_session_fn,
        }
