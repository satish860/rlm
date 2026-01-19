"""
RLM Structured Extraction - LLM-based extraction with Pydantic models.

Provides:
- llm_extract(): Extract structured data from document sections
- llm_extract_parallel(): Parallel extraction across multiple sections
"""

from typing import List, Tuple, Type, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel

from rlm.providers.base import BaseProvider


def llm_extract(
    provider: BaseProvider,
    prompt: str,
    response_model: Type[BaseModel],
    model: str,
    pages: List[str] = None,
    start_page: int = None,
    end_page: int = None,
    progress_callback: Callable[[str], None] = None,
    timeout: float = 120.0
) -> Any:
    """
    Extract structured data using LLM with Pydantic model.

    Args:
        provider: LLM provider instance
        prompt: Extraction prompt
        response_model: Pydantic model class (or List[Model])
        model: Model identifier
        pages: List of page texts (1-indexed via start_page/end_page)
        start_page: Start page number (1-indexed)
        end_page: End page number (1-indexed)
        progress_callback: Function called with progress messages
        timeout: Request timeout in seconds

    Returns:
        Instance(s) of response_model with extracted data

    Example:
        class Contact(BaseModel):
            name: str
            phone: str = None
            email: str = None
            page: int

        # Extract from pages 1-5
        contacts = llm_extract(
            provider,
            "Extract all contact entries",
            List[Contact],
            "openai/gpt-4o-mini",
            pages=pages,
            start_page=1,
            end_page=5
        )
    """
    def _progress(msg: str):
        if progress_callback:
            progress_callback(msg)

    # Build full prompt with page content if specified
    if pages and start_page and end_page:
        page_count = end_page - start_page + 1
        if page_count > 5:
            _progress(f"WARNING: Large page range ({page_count} pages). Consider smaller chunks.")

        # Get section content (convert to 0-indexed)
        start_idx = max(0, start_page - 1)
        end_idx = min(len(pages), end_page)
        content = '\n\n'.join(pages[start_idx:end_idx])
        content_size = len(content)

        full_prompt = f"""{prompt}

DOCUMENT SECTION (pages {start_page}-{end_page}):
{content}"""
    else:
        full_prompt = prompt
        content_size = len(full_prompt)

    _progress(f"llm_extract: {prompt[:50]}... ({content_size} chars)")

    try:
        response = provider.extract(
            full_prompt,
            response_model,
            model,
            timeout=timeout
        )
        _progress("llm_extract: Done")
        return response
    except Exception as e:
        _progress(f"llm_extract ERROR: {e}")
        raise


def llm_extract_parallel(
    provider: BaseProvider,
    sections: List[Tuple[int, int, str]],
    prompt_template: str,
    response_model: Type[BaseModel],
    model: str,
    pages: List[str],
    max_workers: int = 5,
    progress_callback: Callable[[str], None] = None,
    timeout: float = 120.0
) -> List[Tuple[str, int, int, Any, Optional[str]]]:
    """
    Extract from multiple sections in parallel.

    Much faster than sequential llm_extract calls for multi-section documents.

    Args:
        provider: LLM provider instance
        sections: List of (start_page, end_page, category) tuples
        prompt_template: Prompt with {category} placeholder
        response_model: Pydantic model (e.g., List[Contact])
        model: Model identifier
        pages: List of page texts
        max_workers: Max concurrent API calls (default 5)
        progress_callback: Function called with progress messages
        timeout: Request timeout per section

    Returns:
        List of (category, start_page, end_page, results, error) tuples

    Example:
        sections = [
            (1, 5, "Companies"),
            (6, 10, "NGOs"),
            (11, 15, "Seeds")
        ]
        results = llm_extract_parallel(
            provider,
            sections,
            "Extract all contacts from {category}",
            List[Contact],
            "openai/gpt-4o-mini",
            pages,
            max_workers=5
        )
        for category, start, end, items, error in results:
            if not error:
                for item in items:
                    records.append(item.model_dump())
    """
    def _progress(msg: str):
        if progress_callback:
            progress_callback(msg)

    def extract_one(section: Tuple[int, int, str]):
        start, end, category = section
        prompt = prompt_template.format(category=category)
        try:
            result = llm_extract(
                provider,
                prompt,
                response_model,
                model,
                pages=pages,
                start_page=start,
                end_page=end,
                progress_callback=progress_callback,
                timeout=timeout
            )
            return (category, start, end, result, None)
        except Exception as e:
            return (category, start, end, None, str(e))

    _progress(f"Starting parallel extraction of {len(sections)} sections (max {max_workers} workers)...")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_one, s): s for s in sections}

        for future in as_completed(futures):
            category, start, end, data, error = future.result()
            if error:
                _progress(f"FAILED: {category} (pages {start}-{end}): {error}")
                results.append((category, start, end, [], error))
            else:
                count = len(data) if isinstance(data, list) else 1
                _progress(f"DONE: {category} (pages {start}-{end}): {count} records")
                results.append((category, start, end, data, None))

    _progress(f"Parallel extraction complete: {len(results)} sections")
    return results


def get_section(pages: List[str], start_page: int, end_page: int, padding: int = 1) -> str:
    """
    Get content for pages start_page to end_page (1-indexed) with optional padding.

    Args:
        pages: List of page texts
        start_page: Start page number (1-indexed)
        end_page: End page number (1-indexed)
        padding: Extra pages to include before/after

    Returns:
        Combined text of specified pages
    """
    start = max(0, start_page - 1 - padding)
    end = min(len(pages), end_page + padding)
    return '\n\n'.join(pages[start:end])


def ask_about_section(
    provider: BaseProvider,
    question: str,
    pages: List[str],
    start_page: int,
    end_page: int,
    model: str,
    progress_callback: Callable[[str], None] = None
) -> str:
    """
    Ask a question about a specific document section.

    Args:
        provider: LLM provider instance
        question: Question to ask
        pages: List of page texts
        start_page: Start page number (1-indexed)
        end_page: End page number (1-indexed)
        model: Model identifier
        progress_callback: Function called with progress messages

    Returns:
        Answer text from LLM
    """
    content = get_section(pages, start_page, end_page)
    prompt = f"""Answer this question about the following document section:

QUESTION: {question}

DOCUMENT SECTION (pages {start_page}-{end_page}):
{content}"""

    if progress_callback:
        progress_callback(f"ask_about_section: {question[:50]}...")

    response = provider.chat(
        messages=[{"role": "user", "content": prompt}],
        model=model
    )

    return response["choices"][0]["message"]["content"]
