import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import instructor

load_dotenv()

client = instructor.from_openai(
    OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
)

MODEL = "openai/gpt-4o-mini"


class PageRange(BaseModel):
    start: int = Field(..., description="Starting page number")
    end: int = Field(..., description="Ending page number")


class Segment(BaseModel):
    heading: str = Field(..., description="The heading or title of the section")
    description: str = Field(..., description="A brief description of the section")
    page_range: PageRange = Field(..., description="Page range for this section")


def split_into_pages(text: str, lines_per_page: int = 50) -> list[str]:
    """Split text into pages by line count."""
    lines = text.split('\n')
    pages = []
    for i in range(0, len(lines), lines_per_page):
        page_lines = lines[i:i + lines_per_page]
        page_text = f"### Page Number: [PG:{i // lines_per_page + 1}]\n" + '\n'.join(page_lines)
        pages.append(page_text)
    return pages


def process_chunk(chunk_data: tuple) -> list[dict]:
    """Process a chunk and return segments."""
    chunk_index, text_chunk, start_page, end_page = chunk_data

    SYSTEM_PROMPT = """You are a document segmentation AI.
Analyze the document and divide it into logical sections based on content and structure.

For each section provide:
1. A heading/title
2. A brief description
3. The page range where it appears

CRITICAL:
- Use the page numbers from "### Page Number: [PG:X]" markers
- Ensure ALL pages from start to end are covered
- No gaps in page coverage"""

    USER_PROMPT = f"""Segment this document (pages {start_page} to {end_page}).

<Document>
{text_chunk}
</Document>

Return a list of segments covering ALL pages from {start_page} to {end_page}."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT}
            ],
            response_model=list[Segment]
        )

        return [
            {
                "heading": seg.heading,
                "description": seg.description,
                "page_range": {"start": seg.page_range.start, "end": seg.page_range.end}
            }
            for seg in response
        ]
    except Exception as e:
        print(f"Error processing chunk {chunk_index}: {e}")
        return []


def segment_document(text: str, lines_per_page: int = 50, chunk_size: int = 10, max_workers: int = 5) -> list[dict]:
    """Segment a document into sections with page ranges."""

    pages = split_into_pages(text, lines_per_page)
    print(f"Split into {len(pages)} pages")

    chunks = []
    for i in range(0, len(pages), chunk_size):
        chunk_pages = pages[i:i + chunk_size]
        text_chunk = '\n\n'.join(chunk_pages)
        start_page = i + 1
        end_page = min(i + chunk_size, len(pages))
        chunks.append((i, text_chunk, start_page, end_page))

    print(f"Created {len(chunks)} chunks for parallel processing")

    all_segments = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_chunk, chunk): chunk[0] for chunk in chunks}

        with tqdm(total=len(chunks), desc="Segmenting") as pbar:
            for future in as_completed(futures):
                segments = future.result()
                all_segments.extend(segments)
                pbar.update(1)

    all_segments.sort(key=lambda x: x["page_range"]["start"])

    return all_segments


def get_section_content(pages: list[str], segment: dict) -> str:
    """Extract content for a segment from pages."""
    start = segment["page_range"]["start"] - 1
    end = segment["page_range"]["end"]
    return '\n\n'.join(pages[start:end])


if __name__ == "__main__":
    with open("recursive_language_models.converted.md", "r", encoding="utf-8") as f:
        text = f.read()

    segments = segment_document(text, lines_per_page=50, chunk_size=10, max_workers=3)

    print(f"\n=== Found {len(segments)} segments ===\n")
    for seg in segments:
        print(f"[{seg['page_range']['start']}-{seg['page_range']['end']}] {seg['heading']}")
        print(f"    {seg['description'][:80]}...")
        print()
