"""
RLM Command Line Interface - Extract, query, and visualize documents.

Usage:
    rlm extract <document> --schema <name> --output <path>
    rlm query <document> "<question>"
    rlm visualize <result.json> --output <report.html>
    rlm sessions list
    rlm sessions info <name>
"""

import sys
import json
import csv
from pathlib import Path
from typing import Optional

import click

import rlm
from rlm.core.types import ExtractionResult, Citation
from rlm.reasoning.session import SessionManager


# Schema name to class mapping
BUILTIN_SCHEMAS = {
    "contact": "rlm.extraction.schemas:Contact",
    "invoice": "rlm.extraction.schemas:Invoice",
    "entity": "rlm.extraction.schemas:Entity",
    "table": "rlm.extraction.schemas:TableRow",
    "keyvalue": "rlm.extraction.schemas:KeyValuePair",
    "financial": "rlm.extraction.schemas:FinancialFigure",
}


def load_schema(schema_spec: str):
    """
    Load a schema class from specification.

    Args:
        schema_spec: Either a builtin name (contact, invoice, etc.)
                    or a module path (path/to/file.py:ClassName)

    Returns:
        Pydantic model class
    """
    # Check builtin schemas first
    if schema_spec.lower() in BUILTIN_SCHEMAS:
        schema_spec = BUILTIN_SCHEMAS[schema_spec.lower()]

    if ":" in schema_spec:
        # Module path format: module.path:ClassName or file.py:ClassName
        module_path, class_name = schema_spec.rsplit(":", 1)

        if module_path.endswith(".py"):
            # Load from file
            import importlib.util
            spec = importlib.util.spec_from_file_location("custom_schema", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            # Import from module path
            import importlib
            module = importlib.import_module(module_path)

        return getattr(module, class_name)
    else:
        raise click.BadParameter(
            f"Unknown schema: {schema_spec}. "
            f"Use builtin names ({', '.join(BUILTIN_SCHEMAS.keys())}) "
            f"or module:ClassName format."
        )


def write_output(data: list, output_path: str, format: str):
    """Write data to file in specified format."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    elif format == "csv":
        if not data:
            path.write_text("")
            return

        # Get all keys from all records
        all_keys = set()
        for record in data:
            all_keys.update(record.keys())
        fieldnames = sorted(all_keys)

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in data:
                # Flatten nested objects
                flat = {}
                for k, v in record.items():
                    if isinstance(v, (dict, list)):
                        flat[k] = json.dumps(v, default=str)
                    else:
                        flat[k] = v
                writer.writerow(flat)


@click.group()
@click.version_option(version=rlm.__version__, prog_name="rlm")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx, verbose):
    """RLM - Recursive Language Model for document extraction."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@cli.command()
@click.argument("document", type=click.Path(exists=True))
@click.option("--schema", "-s", default=None,
              help="Schema name (contact, invoice, entity, table) or module:ClassName")
@click.option("--output", "-o", default=None,
              help="Output file path (default: stdout)")
@click.option("--format", "-f", type=click.Choice(["json", "csv"]), default="json",
              help="Output format")
@click.option("--max-iterations", "-i", type=int, default=40,
              help="Maximum iterations")
@click.option("--model", "-m", default=None,
              help="Root model (e.g., anthropic/claude-sonnet-4.5)")
@click.option("--visualize", is_flag=True,
              help="Generate HTML visualization")
@click.pass_context
def extract(ctx, document, schema, output, format, max_iterations, model, visualize):
    """
    Extract structured data from a document.

    Examples:
        rlm extract invoice.pdf --schema invoice -o result.json
        rlm extract contacts.pdf --schema contact --format csv -o contacts.csv
        rlm extract doc.pdf --schema myschema.py:MyModel
    """
    verbose = ctx.obj.get("verbose", False)

    # Load schema if specified
    schema_class = None
    if schema:
        try:
            schema_class = load_schema(schema)
            if verbose:
                click.echo(f"Using schema: {schema_class.__name__}")
        except Exception as e:
            raise click.ClickException(f"Failed to load schema: {e}")

    click.echo(f"Extracting from: {document}")

    try:
        result = rlm.extract(
            document,
            schema=schema_class,
            verbose=verbose,
            max_iterations=max_iterations,
            root_model=model
        )

        click.echo(f"Extracted {len(result.data)} records in {result.iterations} iterations")

        # Output results
        if output:
            write_output(result.data, output, format)
            click.echo(f"Saved to: {output}")
        else:
            # Print to stdout
            if format == "json":
                click.echo(json.dumps(result.data, indent=2, ensure_ascii=False, default=str))
            else:
                # CSV to stdout
                if result.data:
                    all_keys = set()
                    for record in result.data:
                        all_keys.update(record.keys())
                    fieldnames = sorted(all_keys)
                    click.echo(",".join(fieldnames))
                    for record in result.data:
                        row = [str(record.get(k, "")) for k in fieldnames]
                        click.echo(",".join(row))

        # Generate visualization if requested
        if visualize:
            html_output = output.replace(".json", ".html") if output else "report.html"
            rlm.visualize(result, output=html_output)
            click.echo(f"Visualization saved to: {html_output}")

    except Exception as e:
        raise click.ClickException(f"Extraction failed: {e}")


@cli.command()
@click.argument("document", type=click.Path(exists=True))
@click.argument("question")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--model", "-m", default=None, help="Root model")
@click.option("--max-iterations", "-i", type=int, default=20, help="Maximum iterations")
@click.pass_context
def query(ctx, document, question, as_json, model, max_iterations):
    """
    Ask a question about a document.

    Examples:
        rlm query report.pdf "What was Q3 revenue?"
        rlm query contract.pdf "When does the contract expire?" --json
    """
    verbose = ctx.obj.get("verbose", False)

    click.echo(f"Querying: {document}")
    click.echo(f"Question: {question}")

    try:
        result = rlm.query(
            document,
            question,
            verbose=verbose,
            max_iterations=max_iterations,
            root_model=model
        )

        if as_json:
            output = {
                "question": question,
                "answer": result.answer,
                "confidence": result.confidence,
                "citations": [
                    {"text": c.snippet, "page": c.page, "note": c.note}
                    for c in result.citations
                ]
            }
            click.echo(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            click.echo("")
            click.echo(f"Answer: {result.answer}")
            click.echo(f"Confidence: {result.confidence:.0%}")
            if result.citations:
                click.echo("")
                click.echo("Citations:")
                for c in result.citations:
                    click.echo(f"  - Page {c.page}: \"{c.snippet[:60]}...\"")

    except Exception as e:
        raise click.ClickException(f"Query failed: {e}")


@cli.command("visualize")
@click.argument("result_file", type=click.Path(exists=True))
@click.option("--output", "-o", default="report.html", help="Output HTML file")
@click.option("--open", "open_browser", is_flag=True, help="Open in browser")
@click.pass_context
def visualize_cmd(ctx, result_file, output, open_browser):
    """
    Generate HTML visualization from extraction result.

    Examples:
        rlm visualize result.json -o report.html
        rlm visualize result.json --open
    """
    try:
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both raw data array and full result object
        if isinstance(data, list):
            # Raw data array
            result = ExtractionResult(
                data=data,
                citations=[],
                thinking_log=[],
                confidence_history=[],
                verification={},
                iterations=0,
                document_path=result_file
            )
        elif isinstance(data, dict):
            # Full result object
            citations = [
                Citation(
                    snippet=c.get("snippet", c.get("text", "")),
                    page=c.get("page", 0),
                    note=c.get("note", c.get("context", ""))
                )
                for c in data.get("citations", [])
            ]
            result = ExtractionResult(
                data=data.get("data", []),
                citations=citations,
                thinking_log=data.get("thinking_log", []),
                confidence_history=data.get("confidence_history", []),
                verification=data.get("verification", {}),
                iterations=data.get("iterations", 0),
                document_path=data.get("document_path", result_file)
            )
        else:
            raise click.ClickException("Invalid result file format")

        rlm.visualize(result, output=output, open_browser=open_browser)
        click.echo(f"Visualization saved to: {output}")

    except json.JSONDecodeError as e:
        raise click.ClickException(f"Invalid JSON file: {e}")
    except Exception as e:
        raise click.ClickException(f"Visualization failed: {e}")


@cli.group()
def sessions():
    """Manage extraction sessions."""
    pass


@sessions.command("list")
def sessions_list():
    """List saved sessions."""
    manager = SessionManager()
    click.echo(manager.format_sessions_table())


@sessions.command("info")
@click.argument("name")
def sessions_info(name):
    """Show details about a session."""
    manager = SessionManager()
    info = manager.get_session_info(name)

    if not info:
        raise click.ClickException(f"Session not found: {name}")

    click.echo(f"Session: {info['name']}")
    click.echo(f"Saved: {info['saved_at']}")
    click.echo(f"Records: {info['records_count']}")
    click.echo(f"Citations: {info['citations_count']}")
    click.echo(f"Thoughts: {info['thoughts_count']}")
    click.echo(f"Pages: {info['total_pages']}")
    click.echo(f"Confidence: {info['final_confidence']:.0%}")

    if info.get("metadata"):
        click.echo("")
        click.echo("Metadata:")
        for k, v in info["metadata"].items():
            click.echo(f"  {k}: {v}")


@sessions.command("delete")
@click.argument("name")
@click.confirmation_option(prompt="Are you sure you want to delete this session?")
def sessions_delete(name):
    """Delete a saved session."""
    manager = SessionManager()
    if manager.delete_session(name):
        click.echo(f"Deleted session: {name}")
    else:
        raise click.ClickException(f"Session not found: {name}")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
