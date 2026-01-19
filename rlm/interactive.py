"""
RLM Interactive CLI - Conversational document extraction for business users.

No code required. Just drop a file and extract data through natural dialogue.

Usage:
    $ rlm

    RLM - Document Extraction

    Drop a file path or type 'help':
    > invoices/january.pdf

    Analyzing...

    Document Type: Invoice (95% confidence)

    Suggested Fields:
      [x] 1. vendor_name
      [x] 2. invoice_number
      ...

    Toggle fields by number, '+name' to add, or Enter to extract:
    > [Enter]

    Extracting... Done!

    Export to: (1) CSV  (2) Excel  (3) JSON  (4) Skip
    > 1

    Saved to: invoices_january_extracted.csv
"""

import sys
import glob
from pathlib import Path
from typing import Optional, List
from enum import Enum

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.text import Text

import rlm.business as biz
from rlm.business import DocumentAnalysis, ExtractionResult


class State(Enum):
    """Interactive session states."""
    WELCOME = "welcome"
    ANALYZING = "analyzing"
    FIELDS = "fields"
    EXTRACTING = "extracting"
    RESULTS = "results"
    EXPORT = "export"
    NEXT = "next"
    EXIT = "exit"


class InteractiveSession:
    """
    Manages conversation state and flow for interactive extraction.

    State machine:
        welcome -> analyzing -> fields -> extracting -> results -> export -> next
                      ^                                                       |
                      |_______________________________________________________|
    """

    def __init__(self):
        self.console = Console()
        self.state = State.WELCOME
        self.document_path: Optional[str] = None
        self.analysis: Optional[DocumentAnalysis] = None
        self.result: Optional[ExtractionResult] = None
        self.files: List[Path] = []

    def run(self):
        """Main conversation loop."""
        self._show_banner()
        self._last_interrupt_time = 0

        while self.state != State.EXIT:
            try:
                if self.state == State.WELCOME:
                    self._handle_welcome()
                elif self.state == State.ANALYZING:
                    self._handle_analyzing()
                elif self.state == State.FIELDS:
                    self._handle_fields()
                elif self.state == State.EXTRACTING:
                    self._handle_extracting()
                elif self.state == State.RESULTS:
                    self._handle_results()
                elif self.state == State.EXPORT:
                    self._handle_export()
                elif self.state == State.NEXT:
                    self._handle_next()
            except KeyboardInterrupt:
                self._handle_interrupt()
            except Exception as e:
                self.console.print(f"\n[red]Error: {e}[/red]\n")
                # Reset to welcome on error
                self.state = State.WELCOME

        self.console.print("\nGoodbye!\n")

    def _handle_interrupt(self):
        """Handle Ctrl+C - double-tap to exit, single to go back."""
        import time
        current_time = time.time()

        # Double Ctrl+C within 1 second = exit
        if current_time - self._last_interrupt_time < 1.0:
            self.console.print("\n")
            self.state = State.EXIT
            return

        self._last_interrupt_time = current_time

        # Single Ctrl+C = go back or show hint
        self.console.print("\n")
        if self.state in (State.EXTRACTING, State.ANALYZING):
            self.console.print("  [yellow]Cancelled.[/yellow]")
            self.state = State.WELCOME
        elif self.state in (State.FIELDS, State.RESULTS, State.EXPORT, State.NEXT):
            self.console.print("  [dim]Press Ctrl+C again to exit, or continue...[/dim]")
            self.state = State.WELCOME
        else:
            self.console.print("  [dim]Press Ctrl+C again to exit.[/dim]")

    def _show_banner(self):
        """Display welcome banner."""
        banner = Text()
        banner.append("\n  RLM - Document Extraction\n", style="bold cyan")
        banner.append("  No code required. Drop a file and extract.\n", style="dim")
        self.console.print(Panel(banner, border_style="cyan"))

    def _handle_welcome(self):
        """Greet user, get document path."""
        self.console.print("\n  Drop a file path or type 'help':")
        user_input = Prompt.ask("  >", default="").strip()

        if not user_input:
            return

        if user_input.lower() == "help":
            self._show_help()
            return

        if user_input.lower() in ("exit", "quit", "q"):
            self.state = State.EXIT
            return

        # Validate file path
        if self._validate_path(user_input):
            self.document_path = user_input
            self.state = State.ANALYZING

    def _validate_path(self, path: str) -> bool:
        """Check if path exists or is valid glob pattern."""
        # Handle glob patterns
        if "*" in path:
            matches = glob.glob(path)
            if not matches:
                self.console.print(f"\n  [red]No files match pattern: {path}[/red]")
                return False
            self.files = [Path(m) for m in matches if Path(m).is_file()]
            if not self.files:
                self.console.print(f"\n  [red]No valid files found matching: {path}[/red]")
                return False
            self.console.print(f"\n  Found {len(self.files)} file(s)")
            return True

        # Single file
        p = Path(path)
        if not p.exists():
            self.console.print(f"\n  [red]File not found: {path}[/red]")
            return False
        if not p.is_file():
            self.console.print(f"\n  [red]Not a file: {path}[/red]")
            return False

        self.files = [p]
        return True

    def _handle_analyzing(self):
        """Analyze document and detect fields."""
        self.console.print("")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True
        ) as progress:
            task = progress.add_task("Analyzing document(s)...", total=None)

            try:
                self.analysis = biz.discover(
                    self.document_path,
                    verbose=False
                )
            except Exception as e:
                self.console.print(f"\n  [red]Analysis failed: {e}[/red]")
                self.state = State.WELCOME
                return

        # Show analysis results
        conf_pct = int(self.analysis.confidence * 100)
        self.console.print(f"\n  Document Type: [cyan]{self.analysis.document_type}[/cyan] ({conf_pct}% confidence)")
        self.console.print(f"  Files: {self.analysis.file_count}, Pages: {self.analysis.total_pages}")

        self.state = State.FIELDS

    def _handle_fields(self):
        """Show fields, let user toggle."""
        self._display_fields()

        self.console.print("\n  Toggle fields by number, '+name' to add custom, or Enter to extract:")
        user_input = Prompt.ask("  >", default="").strip()

        if not user_input:
            # Proceed to extraction
            enabled = self.analysis.get_enabled_fields()
            if not enabled:
                self.console.print("\n  [yellow]No fields enabled. Enable at least one field.[/yellow]")
                return
            self.state = State.EXTRACTING
            return

        if user_input.lower() == "back":
            self.state = State.WELCOME
            return

        if user_input.lower() == "help":
            self._show_field_help()
            return

        # Toggle field by number
        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(self.analysis.suggested_fields):
                field = self.analysis.suggested_fields[idx]
                field.enabled = not field.enabled
                status = "enabled" if field.enabled else "disabled"
                self.console.print(f"\n  {field.name}: {status}")
            else:
                self.console.print(f"\n  [red]Invalid field number: {user_input}[/red]")
            return

        # Toggle multiple fields (e.g., "1 2 3" or "1,2,3")
        parts = user_input.replace(",", " ").split()
        if all(p.isdigit() for p in parts):
            for p in parts:
                idx = int(p) - 1
                if 0 <= idx < len(self.analysis.suggested_fields):
                    field = self.analysis.suggested_fields[idx]
                    field.enabled = not field.enabled
            self.console.print("\n  Fields toggled")
            return

        # Add custom field with +name
        if user_input.startswith("+"):
            field_name = user_input[1:].strip().replace(" ", "_").lower()
            if field_name:
                self.analysis.add_field(
                    name=field_name,
                    description=f"Custom field: {field_name}",
                    field_type="text"
                )
                self.console.print(f"\n  Added custom field: [green]{field_name}[/green]")
            return

        self.console.print(f"\n  [dim]Unknown command. Type 'help' for options.[/dim]")

    def _display_fields(self):
        """Show fields with checkboxes and examples."""
        self.console.print("\n  [bold]Suggested Fields:[/bold]\n")

        for i, field in enumerate(self.analysis.suggested_fields, 1):
            marker = "[green][x][/green]" if field.enabled else "[ ]"

            # Format examples
            examples_str = ""
            if field.examples:
                examples = [f'"{ex}"' for ex in field.examples[:2]]
                examples_str = f"  [dim]{', '.join(examples)}[/dim]"

            # Format field type
            type_str = f"[dim]({field.field_type})[/dim]" if field.field_type != "text" else ""

            self.console.print(f"    {marker} {i:2}. {field.name:20} {type_str}{examples_str}")

    def _handle_extracting(self):
        """Run extraction with verbose output so user sees LLM activity."""
        enabled_fields = self.analysis.get_enabled_fields()

        self.console.print(f"\n  Extracting {len(enabled_fields)} fields from {len(self.files)} file(s)...")
        self.console.print("  [dim]LLM is analyzing the document. Press Ctrl+C to cancel.[/dim]")
        self.console.print("")
        self.console.print("  " + "-" * 60)

        try:
            # Run with verbose=True so user sees real LLM progress
            self.result = biz.extract_simple(
                files=self.document_path,
                fields=enabled_fields,
                merge=True,
                verbose=True  # Show real progress!
            )
        except KeyboardInterrupt:
            self.console.print("\n  " + "-" * 60)
            self.console.print("\n  [yellow]Extraction cancelled.[/yellow]")
            self.state = State.FIELDS
            return
        except Exception as e:
            self.console.print(f"\n  [red]Extraction failed: {e}[/red]")
            self.state = State.FIELDS
            return

        self.console.print("  " + "-" * 60)
        self.console.print(f"\n  [green]Done![/green] Found {self.result.total_records} record(s)\n")
        self.state = State.RESULTS

    def _handle_results(self):
        """Display extraction results."""
        if not self.result or not self.result.records:
            self.console.print("  [yellow]No records extracted.[/yellow]")
            self.state = State.NEXT
            return

        self._display_results_table()

        # Show issues if any
        if self.result.issues:
            self.console.print(f"\n  [yellow]Issues: {len(self.result.issues)} file(s) had errors[/yellow]")

        self.state = State.EXPORT

    def _display_results_table(self):
        """Show extraction results as table."""
        if not self.result.records:
            return

        # Create table
        table = Table(show_header=True, header_style="bold cyan", box=None)

        # Add columns (limit to first 5 visible columns)
        visible_fields = [f for f in self.result.fields if not f.startswith("_")][:5]
        for col in visible_fields:
            table.add_column(col, max_width=25)

        # Add rows (limit to first 10)
        for record in self.result.records[:10]:
            row = []
            for field in visible_fields:
                val = record.get(field, "")
                if val is None:
                    val = ""
                val_str = str(val)[:25]  # Truncate long values
                row.append(val_str)
            table.add_row(*row)

        self.console.print(table)

        # Show if truncated
        if len(self.result.records) > 10:
            self.console.print(f"\n  [dim]Showing 10 of {len(self.result.records)} records[/dim]")

    def _handle_export(self):
        """Export results to file."""
        self.console.print("\n  Export to: (1) CSV  (2) Excel  (3) JSON  (4) Skip")
        choice = Prompt.ask("  >", default="1").strip()

        if choice == "4" or choice.lower() == "skip":
            self.state = State.NEXT
            return

        # Generate default filename
        base_name = Path(self.files[0]).stem if self.files else "extracted"

        try:
            if choice == "1" or choice.lower() == "csv":
                output_path = f"{base_name}_extracted.csv"
                self.result.to_csv(output_path)
                self.console.print(f"\n  [green]Saved to: {output_path}[/green]")

            elif choice == "2" or choice.lower() == "excel":
                output_path = f"{base_name}_extracted.xlsx"
                try:
                    self.result.to_excel(output_path)
                    self.console.print(f"\n  [green]Saved to: {output_path}[/green]")
                except ImportError:
                    self.console.print("\n  [red]Excel export requires openpyxl: pip install openpyxl[/red]")
                    return

            elif choice == "3" or choice.lower() == "json":
                output_path = f"{base_name}_extracted.json"
                self.result.to_json(output_path)
                self.console.print(f"\n  [green]Saved to: {output_path}[/green]")

            else:
                self.console.print(f"\n  [dim]Unknown option: {choice}[/dim]")
                return

        except Exception as e:
            self.console.print(f"\n  [red]Export failed: {e}[/red]")
            return

        self.state = State.NEXT

    def _handle_next(self):
        """Prompt for next action."""
        self.console.print("\n  What next?")
        self.console.print("  (1) Extract from another file")
        self.console.print("  (2) Add more fields and re-extract")
        self.console.print("  (3) View citations/evidence")
        self.console.print("  (4) Exit")

        choice = Prompt.ask("  >", default="4").strip()

        if choice == "1":
            # Reset for new file
            self.document_path = None
            self.analysis = None
            self.result = None
            self.files = []
            self.state = State.WELCOME

        elif choice == "2":
            if self.analysis:
                self.state = State.FIELDS
            else:
                self.console.print("\n  [yellow]No analysis available. Start with a new file.[/yellow]")
                self.state = State.WELCOME

        elif choice == "3":
            self._show_citations()

        elif choice == "4" or choice.lower() in ("exit", "quit", "q"):
            self.state = State.EXIT

        else:
            self.console.print(f"\n  [dim]Unknown option: {choice}[/dim]")

    def _show_citations(self):
        """Display citations/evidence for extracted data."""
        if not self.result or not self.result.citations:
            self.console.print("\n  [yellow]No citations available.[/yellow]")
            return

        self.console.print("\n  [bold]Citations/Evidence:[/bold]\n")

        for i, cite in enumerate(self.result.citations[:10], 1):
            file_name = cite.get("file", "")
            page = cite.get("page", "?")
            snippet = cite.get("snippet", "")[:100]

            self.console.print(f"  {i}. [cyan]{file_name}[/cyan] (page {page})")
            self.console.print(f"     [dim]\"{snippet}...\"[/dim]\n")

        if len(self.result.citations) > 10:
            self.console.print(f"  [dim]Showing 10 of {len(self.result.citations)} citations[/dim]")

    def _show_help(self):
        """Show help text."""
        help_text = """
  [bold]RLM Interactive Help[/bold]

  [cyan]Getting Started:[/cyan]
    1. Enter a file path (e.g., invoice.pdf, documents/*.pdf)
    2. Review suggested fields and toggle as needed
    3. Press Enter to extract
    4. Export to CSV, Excel, or JSON

  [cyan]Commands:[/cyan]
    help      - Show this help
    exit/quit - Exit the program
    back      - Go back to previous step

  [cyan]Field Selection:[/cyan]
    1-9       - Toggle field by number
    1 2 3     - Toggle multiple fields
    +name     - Add custom field (e.g., +po_number)
    Enter     - Start extraction

  [cyan]Supported File Types:[/cyan]
    PDF, TXT, MD, DOCX

  [cyan]Glob Patterns:[/cyan]
    *.pdf           - All PDFs in current directory
    invoices/*.pdf  - All PDFs in invoices folder
    **/*.pdf        - All PDFs recursively
"""
        self.console.print(help_text)

    def _show_field_help(self):
        """Show field selection help."""
        help_text = """
  [bold]Field Selection Help[/bold]

  [cyan]Toggle Fields:[/cyan]
    Enter a number to toggle that field on/off
    Example: 1 (toggles first field)

  [cyan]Toggle Multiple:[/cyan]
    Enter multiple numbers separated by spaces
    Example: 1 3 5 (toggles fields 1, 3, and 5)

  [cyan]Add Custom Field:[/cyan]
    Use + prefix to add a new field
    Example: +purchase_order (adds purchase_order field)

  [cyan]Continue:[/cyan]
    Press Enter with no input to start extraction

  [cyan]Go Back:[/cyan]
    Type 'back' to return to file selection
"""
        self.console.print(help_text)



def main():
    """Entry point for interactive CLI."""
    session = InteractiveSession()
    session.run()


if __name__ == "__main__":
    main()
