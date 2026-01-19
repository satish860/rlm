"""
View saved session data from previous extractions.

Usage:
    python examples/view_session.py                    # List sessions
    python examples/view_session.py <session_name>    # View specific session
"""

import sys
import json
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def list_sessions():
    """List all saved sessions."""
    sessions_dir = Path(__file__).parent.parent / "sessions"

    if not sessions_dir.exists():
        print("No sessions folder found")
        return

    sessions = list(sessions_dir.glob("*.json"))
    if not sessions:
        print("No sessions found")
        return

    print(f"Found {len(sessions)} sessions:\n")
    for path in sorted(sessions, key=lambda p: p.stat().st_mtime, reverse=True):
        size_kb = path.stat().st_size / 1024
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            records = len(data.get("records", []))
            citations = len(data.get("citations", []))
            print(f"  {path.stem}")
            print(f"    Records: {records}, Citations: {citations}, Size: {size_kb:.1f}KB")
        except:
            print(f"  {path.stem} ({size_kb:.1f}KB)")
        print()


def view_session(name: str):
    """View a specific session."""
    sessions_dir = Path(__file__).parent.parent / "sessions"

    # Try with and without .json extension
    path = sessions_dir / f"{name}.json"
    if not path.exists():
        path = sessions_dir / name
    if not path.exists():
        print(f"Session not found: {name}")
        list_sessions()
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records", [])
    citations = data.get("citations", [])

    print(f"Session: {path.stem}")
    print(f"Records: {len(records)}")
    print(f"Citations: {len(citations)}")
    print("=" * 60)

    # Show sample records
    print(f"\nSample Records (first 10):\n")
    for i, record in enumerate(records[:10], 1):
        name = record.get("name") or record.get("company") or "N/A"
        company = record.get("company", "")
        phone = record.get("phone") or record.get("phone_office") or ""
        email = record.get("email", "")
        page = record.get("page", "?")

        print(f"[{i}] {name}")
        if company and company != name:
            print(f"    Company: {company}")
        if phone:
            print(f"    Phone: {phone[:50]}")
        if email:
            print(f"    Email: {email}")
        print(f"    Page: {page}")
        print()

    if len(records) > 10:
        print(f"... and {len(records) - 10} more records")

    # Show categories if available
    categories = {}
    for r in records:
        cat = r.get("category", "Unknown")
        categories[cat] = categories.get(cat, 0) + 1

    if categories:
        print(f"\nCategories:")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")

    # Export option
    print(f"\nTo export to CSV:")
    print(f"  python -c \"import json,csv; d=json.load(open('{path}')); w=csv.DictWriter(open('export.csv','w',newline=''),d['records'][0].keys()); w.writeheader(); w.writerows(d['records'])\"")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        view_session(sys.argv[1])
    else:
        list_sessions()
