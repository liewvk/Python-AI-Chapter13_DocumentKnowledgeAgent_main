"""Read text, Markdown, PDF, and DOCX knowledge documents."""
import re
from pathlib import Path

import pymupdf
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from settings import (
    ALLOWED_EXTENSIONS, CHUNK_OVERLAP_WORDS, CHUNK_WORDS,
    KNOWLEDGE_DIR, MAX_DOCUMENT_CHARS, MAX_FILE_BYTES,
    MAX_FILES, MAX_PDF_PAGES,
)


def list_documents(folder=KNOWLEDGE_DIR):
    """List supported files in the designated knowledge directory."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        (p for p in folder.iterdir()
         if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS),
        key=lambda p: p.name.lower(),
    )
    if len(paths) > MAX_FILES:
        raise ValueError(f"Limit exceeded: at most {MAX_FILES} documents.")
    for path in paths:
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"Document too large (4 MB limit): {path.name}")
    return paths


def normalize(text):
    """Remove nulls and extra spaces while keeping line breaks."""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def read_document(path):
    """Return ordered sections with meaningful source locations."""
    path = Path(path)
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported format. Use .txt, .md, .pdf, or .docx.")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"File exceeds size limit: {path.name}")

    sections = []
    kind = path.suffix.lower()
    if kind in (".txt", ".md"):
        text = normalize(path.read_text(encoding="utf-8-sig"))
        if text:
            sections.append({"location": "Text", "text": text})

    elif kind == ".pdf":
        with pymupdf.open(path) as pdf:
            if pdf.needs_pass:
                raise ValueError(f"Password-protected PDF: {path.name}")
            if len(pdf) > MAX_PDF_PAGES:
                raise ValueError(f"PDF exceeds {MAX_PDF_PAGES} pages: {path.name}")
            for number, page in enumerate(pdf, start=1):
                text = normalize(page.get_text("text", sort=True))
                if text:
                    sections.append({"location": f"Page {number}", "text": text})

    else:  # .docx
        document = Document(str(path))
        paragraph_number = 0
        table_number = 0
        for block in document.iter_inner_content():
            if isinstance(block, Paragraph):
                paragraph_number += 1
                text = normalize(block.text)
                if text:
                    sections.append({
                        "location": f"Paragraph {paragraph_number}", "text": text
                    })
            elif isinstance(block, Table):
                table_number += 1
                lines = []
                for row in block.rows:
                    cells = [normalize(cell.text).replace("\n", "; ")
                             for cell in row.cells]
                    if any(cells):
                        lines.append(" | ".join(cells))
                if lines:
                    sections.append({
                        "location": f"Table {table_number}",
                        "text": "\n".join(lines),
                    })

    if not sections:
        reason = (" This may be a scanned or image-only PDF; OCR is not "
                  "included in this project." if kind == ".pdf" else "")
        raise ValueError(f"No extractable text in {path.name}.{reason}")
    if sum(len(section["text"]) for section in sections) > MAX_DOCUMENT_CHARS:
        raise ValueError(f"Extracted text exceeds size limit: {path.name}")
    return sections


def split_into_chunks(text, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    """Split by word count; keep small overlaps between consecutive chunks."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Chunk size must be greater than the nonnegative overlap.")
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def extract_chunks(folder=KNOWLEDGE_DIR):
    """Make source-traceable chunk records from all supported documents."""
    paths = list_documents(folder)
    if not paths:
        raise ValueError("No .pdf, .docx, .txt or .md files found in knowledge/.")
    records = []
    for doc_num, path in enumerate(paths, start=1):
        section_chunks = 0
        for section in read_document(path):
            for part, text in enumerate(split_into_chunks(section["text"]), 1):
                section_chunks += 1
                records.append({
                    "id": f"D{doc_num:02d}-C{section_chunks:03d}",
                    "source": path.name,
                    "location": f"{section['location']}, chunk {part}",
                    "text": text,
                })
    if not records:
        raise ValueError("No searchable text was extracted.")
    return paths, records
