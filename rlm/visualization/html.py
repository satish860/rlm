"""
RLM HTML Visualization - Generate interactive HTML reports.

Provides:
- generate_html_report(): Create full extraction report
- Embedded CSS/JS for standalone viewing
- Citation highlighting and linking
- Collapsible reasoning trace
"""

import json
import html
from typing import Dict, Any, List, Optional
from datetime import datetime

from rlm.core.types import ExtractionResult, Citation
from rlm.visualization.citations import (
    highlight_citations,
    create_citation_index,
    escape_html
)


# Standalone HTML template with embedded CSS/JS
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RLM Extraction Report</title>
    <style>
        :root {
            --primary: #2563eb;
            --primary-light: #dbeafe;
            --success: #16a34a;
            --warning: #ca8a04;
            --danger: #dc2626;
            --gray-50: #f9fafb;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-300: #d1d5db;
            --gray-600: #4b5563;
            --gray-800: #1f2937;
            --gray-900: #111827;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: var(--gray-800);
            background: var(--gray-50);
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }

        header {
            background: white;
            border-bottom: 1px solid var(--gray-200);
            padding: 1.5rem 2rem;
            margin-bottom: 2rem;
        }

        header h1 {
            font-size: 1.5rem;
            color: var(--gray-900);
        }

        header .meta {
            font-size: 0.875rem;
            color: var(--gray-600);
            margin-top: 0.5rem;
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }

        .stat-card {
            background: white;
            border-radius: 8px;
            padding: 1.25rem;
            border: 1px solid var(--gray-200);
        }

        .stat-card .value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary);
        }

        .stat-card .label {
            font-size: 0.875rem;
            color: var(--gray-600);
        }

        .confidence-bar {
            height: 8px;
            background: var(--gray-200);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 0.5rem;
        }

        .confidence-fill {
            height: 100%;
            background: var(--success);
            transition: width 0.3s;
        }

        .confidence-fill.medium {
            background: var(--warning);
        }

        .confidence-fill.low {
            background: var(--danger);
        }

        .section {
            background: white;
            border-radius: 8px;
            border: 1px solid var(--gray-200);
            margin-bottom: 1.5rem;
        }

        .section-header {
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--gray-200);
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .section-header:hover {
            background: var(--gray-50);
        }

        .section-header h2 {
            font-size: 1.125rem;
            font-weight: 600;
        }

        .section-header .toggle {
            color: var(--gray-600);
            transition: transform 0.2s;
        }

        .section-header.collapsed .toggle {
            transform: rotate(-90deg);
        }

        .section-content {
            padding: 1.5rem;
        }

        .section-content.collapsed {
            display: none;
        }

        /* Data table */
        .data-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
        }

        .data-table th,
        .data-table td {
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--gray-200);
        }

        .data-table th {
            background: var(--gray-50);
            font-weight: 600;
            color: var(--gray-600);
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }

        .data-table tr:hover {
            background: var(--gray-50);
        }

        .data-table .page-num {
            color: var(--primary);
            font-weight: 500;
        }

        /* Citations */
        mark.citation {
            background: var(--primary-light);
            padding: 0.125rem 0.25rem;
            border-radius: 2px;
            cursor: pointer;
        }

        mark.citation:hover {
            background: #bfdbfe;
        }

        .citation-index {
            list-style-position: inside;
            padding: 0;
        }

        .citation-index li {
            padding: 0.75rem 0;
            border-bottom: 1px solid var(--gray-100);
        }

        .citation-snippet {
            font-style: italic;
            color: var(--gray-600);
        }

        .citation-page {
            font-size: 0.875rem;
            color: var(--primary);
            font-weight: 500;
        }

        .citation-note {
            color: var(--gray-600);
            font-size: 0.875rem;
        }

        /* Document viewer */
        .document-content {
            max-height: 500px;
            overflow-y: auto;
            background: var(--gray-50);
            padding: 1rem;
            border-radius: 4px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.8125rem;
            white-space: pre-wrap;
            line-height: 1.5;
        }

        /* Thinking log */
        .thinking-entry {
            padding: 0.75rem 1rem;
            background: var(--gray-50);
            border-radius: 4px;
            margin-bottom: 0.5rem;
            border-left: 3px solid var(--primary);
        }

        .thinking-entry .timestamp {
            font-size: 0.75rem;
            color: var(--gray-600);
        }

        .thinking-entry .thought {
            margin-top: 0.25rem;
        }

        /* Verification */
        .verification-item {
            display: flex;
            align-items: center;
            padding: 0.5rem 0;
        }

        .verification-item .icon {
            width: 20px;
            height: 20px;
            margin-right: 0.75rem;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            color: white;
        }

        .verification-item .icon.pass {
            background: var(--success);
        }

        .verification-item .icon.fail {
            background: var(--danger);
        }

        /* Footer */
        footer {
            text-align: center;
            padding: 2rem;
            color: var(--gray-600);
            font-size: 0.875rem;
        }

        footer a {
            color: var(--primary);
            text-decoration: none;
        }

        /* Print styles */
        @media print {
            body {
                background: white;
            }

            .container {
                max-width: none;
                padding: 0;
            }

            .section {
                break-inside: avoid;
            }

            .section-content.collapsed {
                display: block !important;
            }

            .toggle {
                display: none;
            }
        }
    </style>
</head>
<body>
    <header>
        <h1>RLM Extraction Report</h1>
        <div class="meta">
            <span>Document: {{document_path}}</span>
            <span> | </span>
            <span>Generated: {{generated_at}}</span>
        </div>
    </header>

    <div class="container">
        <!-- Stats -->
        <div class="stats">
            <div class="stat-card">
                <div class="value">{{records_count}}</div>
                <div class="label">Records Extracted</div>
            </div>
            <div class="stat-card">
                <div class="value">{{citations_count}}</div>
                <div class="label">Citations</div>
            </div>
            <div class="stat-card">
                <div class="value">{{iterations}}</div>
                <div class="label">Iterations</div>
            </div>
            <div class="stat-card">
                <div class="value">{{confidence_pct}}%</div>
                <div class="label">Confidence</div>
                <div class="confidence-bar">
                    <div class="confidence-fill {{confidence_class}}" style="width: {{confidence_pct}}%"></div>
                </div>
            </div>
        </div>

        <!-- Extracted Data -->
        <div class="section">
            <div class="section-header" onclick="toggleSection(this)">
                <h2>Extracted Data ({{records_count}} records)</h2>
                <span class="toggle">&#9660;</span>
            </div>
            <div class="section-content">
                {{data_table}}
            </div>
        </div>

        <!-- Citations -->
        <div class="section">
            <div class="section-header" onclick="toggleSection(this)">
                <h2>Citations ({{citations_count}})</h2>
                <span class="toggle">&#9660;</span>
            </div>
            <div class="section-content">
                {{citations_html}}
            </div>
        </div>

        <!-- Document with Highlights -->
        {{#if has_document}}
        <div class="section">
            <div class="section-header collapsed" onclick="toggleSection(this)">
                <h2>Source Document</h2>
                <span class="toggle">&#9660;</span>
            </div>
            <div class="section-content collapsed">
                <div class="document-content">{{document_html}}</div>
            </div>
        </div>
        {{/if}}

        <!-- Thinking Log -->
        {{#if has_thinking}}
        <div class="section">
            <div class="section-header collapsed" onclick="toggleSection(this)">
                <h2>Reasoning Trace ({{thinking_count}} thoughts)</h2>
                <span class="toggle">&#9660;</span>
            </div>
            <div class="section-content collapsed">
                {{thinking_html}}
            </div>
        </div>
        {{/if}}

        <!-- Verification -->
        <div class="section">
            <div class="section-header" onclick="toggleSection(this)">
                <h2>Verification</h2>
                <span class="toggle">&#9660;</span>
            </div>
            <div class="section-content">
                {{verification_html}}
            </div>
        </div>
    </div>

    <footer>
        Generated by <a href="https://github.com/satish860/rlm">RLM</a> - Recursive Language Model Library
    </footer>

    <script>
        function toggleSection(header) {
            header.classList.toggle('collapsed');
            const content = header.nextElementSibling;
            content.classList.toggle('collapsed');
        }
    </script>
</body>
</html>'''


def generate_data_table(data: List[Dict[str, Any]]) -> str:
    """Generate HTML table from extracted data."""
    if not data:
        return "<p>No data extracted.</p>"

    # Get all unique keys from data
    all_keys = set()
    for record in data:
        all_keys.update(record.keys())

    # Prioritize certain keys
    priority_keys = ['name', 'title', 'company', 'email', 'phone', 'page']
    sorted_keys = [k for k in priority_keys if k in all_keys]
    sorted_keys.extend(sorted(k for k in all_keys if k not in priority_keys))

    lines = ['<table class="data-table">']
    lines.append('<thead><tr>')
    for key in sorted_keys:
        lines.append(f'<th>{escape_html(key)}</th>')
    lines.append('</tr></thead>')

    lines.append('<tbody>')
    for record in data[:100]:  # Limit to 100 rows for performance
        lines.append('<tr>')
        for key in sorted_keys:
            value = record.get(key, '')
            if key == 'page':
                lines.append(f'<td class="page-num">{escape_html(str(value))}</td>')
            elif isinstance(value, (dict, list)):
                lines.append(f'<td>{escape_html(json.dumps(value, default=str)[:100])}</td>')
            else:
                lines.append(f'<td>{escape_html(str(value)[:200])}</td>')
        lines.append('</tr>')
    lines.append('</tbody>')
    lines.append('</table>')

    if len(data) > 100:
        lines.append(f'<p class="note">Showing 100 of {len(data)} records</p>')

    return '\n'.join(lines)


def generate_thinking_html(thinking_log: List[Dict[str, Any]]) -> str:
    """Generate HTML for thinking log."""
    if not thinking_log:
        return "<p>No reasoning trace recorded.</p>"

    lines = []
    for entry in thinking_log:
        # Handle both dict and ThinkingEntry objects
        if hasattr(entry, 'timestamp'):
            timestamp = str(entry.timestamp)[:19]
            thought = escape_html(str(entry.thought))
        else:
            timestamp = str(entry.get("timestamp", ""))[:19]
            thought = escape_html(entry.get("thought", ""))
        lines.append(f'''<div class="thinking-entry">
            <div class="timestamp">{timestamp}</div>
            <div class="thought">{thought}</div>
        </div>''')

    return '\n'.join(lines)


def generate_verification_html(verification: Dict[str, Any]) -> str:
    """Generate HTML for verification status."""
    if not verification:
        return "<p>No verification data.</p>"

    lines = []

    # Verification passed
    passed = verification.get("verification_passed", False)
    icon_class = "pass" if passed else "fail"
    icon = "&#10003;" if passed else "&#10007;"
    lines.append(f'''<div class="verification-item">
        <div class="icon {icon_class}">{icon}</div>
        <span>Verification: {"Passed" if passed else "Failed"}</span>
    </div>''')

    # Total records
    total = verification.get("total_records", 0)
    lines.append(f'''<div class="verification-item">
        <div class="icon pass">&#10003;</div>
        <span>Total Records: {total}</span>
    </div>''')

    # Pages processed
    pages = verification.get("pages_processed", [])
    if pages:
        lines.append(f'''<div class="verification-item">
            <div class="icon pass">&#10003;</div>
            <span>Pages Processed: {len(pages)}</span>
        </div>''')

    # Categories
    categories = verification.get("categories_found", {})
    if categories:
        cat_str = ", ".join(f"{k}: {v}" for k, v in categories.items())
        lines.append(f'''<div class="verification-item">
            <div class="icon pass">&#10003;</div>
            <span>Categories: {escape_html(cat_str)}</span>
        </div>''')

    # Notes
    notes = verification.get("notes", "")
    if notes:
        lines.append(f'''<div class="verification-item">
            <div class="icon pass">&#10003;</div>
            <span>Notes: {escape_html(notes)}</span>
        </div>''')

    return '\n'.join(lines)


def generate_html_report(
    result: ExtractionResult,
    document_text: str = None,
    include_document: bool = True
) -> str:
    """
    Generate complete HTML report from extraction result.

    Args:
        result: ExtractionResult from extraction
        document_text: Optional document text for highlighting
        include_document: Whether to include document viewer

    Returns:
        Complete HTML string

    Example:
        html = generate_html_report(result, document_text)
        with open("report.html", "w") as f:
            f.write(html)
    """
    # Calculate confidence
    if result.confidence_history:
        last_entry = result.confidence_history[-1]
        # Handle both dict and ConfidenceEntry objects
        if hasattr(last_entry, 'confidence'):
            confidence = last_entry.confidence
        elif isinstance(last_entry, dict):
            confidence = last_entry.get("confidence", 0)
        else:
            confidence = 0
    else:
        confidence = 0
    confidence_pct = int(confidence * 100)

    if confidence_pct >= 80:
        confidence_class = ""
    elif confidence_pct >= 50:
        confidence_class = "medium"
    else:
        confidence_class = "low"

    # Generate sub-components
    data_table = generate_data_table(result.data)

    citations_dicts = [
        {"snippet": c.snippet, "page": c.page, "note": c.note}
        for c in result.citations
    ]
    citations_html = create_citation_index(citations_dicts)

    # Document with highlights
    document_html = ""
    has_document = False
    if include_document and document_text:
        has_document = True
        document_html = highlight_citations(document_text[:50000], citations_dicts)

    # Thinking log
    thinking_html = generate_thinking_html(result.thinking_log)
    has_thinking = bool(result.thinking_log)

    # Verification
    verification_html = generate_verification_html(result.verification)

    # Build final HTML
    html_content = HTML_TEMPLATE

    # Simple template replacement
    replacements = {
        "{{document_path}}": escape_html(result.document_path or "Unknown"),
        "{{generated_at}}": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "{{records_count}}": str(len(result.data)),
        "{{citations_count}}": str(len(result.citations)),
        "{{iterations}}": str(result.iterations),
        "{{confidence_pct}}": str(confidence_pct),
        "{{confidence_class}}": confidence_class,
        "{{data_table}}": data_table,
        "{{citations_html}}": citations_html,
        "{{document_html}}": document_html,
        "{{thinking_html}}": thinking_html,
        "{{thinking_count}}": str(len(result.thinking_log)),
        "{{verification_html}}": verification_html,
    }

    for key, value in replacements.items():
        html_content = html_content.replace(key, value)

    # Handle conditionals
    if has_document:
        html_content = html_content.replace("{{#if has_document}}", "")
        html_content = html_content.replace("{{/if}}", "")
    else:
        # Remove document section
        import re
        html_content = re.sub(
            r'\{\{#if has_document\}\}.*?\{\{/if\}\}',
            '',
            html_content,
            flags=re.DOTALL
        )

    if has_thinking:
        html_content = html_content.replace("{{#if has_thinking}}", "")
    else:
        html_content = re.sub(
            r'\{\{#if has_thinking\}\}.*?\{\{/if\}\}',
            '',
            html_content,
            flags=re.DOTALL
        )

    return html_content
