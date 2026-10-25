# Chapter 13 — Working with Documents, PDFs, and Knowledge Bases

A complete beginner-friendly **Visual Studio 2026 + Python 3.12 + Ollama** project for a local multi-format Retrieval-Augmented Generation (RAG) assistant.

## Files

- `main.py` — console commands and questions.
- `settings.py` — models, limits, and paths.
- `document_loader.py` — read PDF, DOCX, TXT, Markdown; prepare source-traceable chunks.
- `rag_service.py` — embeddings, JSON vector index, semantic retrieval, citations, stale-index checks.
- `knowledge/` — four fictional sample files, including a real two-page PDF and a Word document with a table.
- `tests/test_rag.py` — eight offline tests; no running Ollama server required.

## Install in Visual Studio 2026

1. Create a **Python Application** project called `Chapter13_DocumentKnowledgeAgent` (not a C# project and not Visual Studio Code).
2. Copy all supplied project files and folders into your Python project. Include the `.pdf` and `.docx` files in `knowledge/`.
3. Open **View → Terminal → Developer PowerShell** (or another PowerShell at the project directory).
4. If you have not created the virtual environment yet:

   ```powershell
   py -3.12 -m venv .venv
   ```

5. Install packages into the **same virtual environment that Visual Studio uses**:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

6. Install both Ollama models **separately**:

   ```powershell
   ollama pull qwen3.5:9b
   ollama pull nomic-embed-text
   ollama list
   ```

   The 404 error `model 'nomic-embed-text' not found` means the embedding model has not been pulled to the Ollama server that your application uses.

7. Ensure Ollama is running; set `main.py` as the Visual Studio startup file. Use **Ctrl+F5**, or use:

   ```powershell
   .\.venv\Scripts\python.exe main.py
   ```

If needed, set the interpreter path in the Python project’s **Properties → Debug** to `...\.venv\Scripts\python.exe`.

## Commands

- `/index` — generate embeddings and persist `data/document_index.json`.
- `/docs` — list supported files under `knowledge/`.
- `/preview admissions_and_fees.pdf` — show the first 1000 extracted characters, with page labels.
- `/preview staff_procedures.docx` — inspect extracted Word paragraphs/tables.
- `/search What is the tuition refund policy?` — display relevant passages and scores.
- `/status` — show document count, chunk count, and index freshness.
- `/help` — show all commands.
- `/exit` — close.
- Or enter a question directly, e.g. `How much is the registration charge?`

## Suggested questions

1. What is the one-time registration charge? (PDF)
2. What is the tuition refund rule? (PDF, page 2)
3. What must visitors do at reception? (Word)
4. What is the response target for a critical network outage? (Word table)
5. How many books may students borrow and for how long? (TXT)
6. What should students do if they forget the portal password? (Markdown)
7. What is the cafeteria lunch menu? (**Not documented**; the assistant should say the evidence is insufficient.)

## Index freshness

After editing, adding, or removing a document in `knowledge/`, run `/index` again. The application checks document SHA-256 hashes and refuses stale searches.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests verify extraction from real PDF and DOCX examples, TXT/Markdown reading, chunk overlap, source labels, index persistence, changed-file detection, retrieval, prompt evidence, and cosine similarity.

## Important limitations

- **PDF images/scans**: the program extracts selectable text only. Image-only or scanned PDFs require OCR, which is **not included**.
- **Protected PDFs**: encrypted/password-protected PDFs are not supported.
- **Word files**: `.docx` is supported; convert legacy `.doc` files first. Text in drawings/text boxes, nested tables, comments, tracked revisions, etc. may not be fully extracted.
- **Search**: cosine-similarity thresholds are heuristics, and source presence does not guarantee generated answers are correct.
- **Privacy**: the files and JSON embeddings stay in the local project, but anyone with local access to them may inspect the information. Never upload confidential data to a public repository.
- **Examples**: all school policies and fees in the sample documents are **fictional educational examples**.

## Dependencies

See `requirements.txt`: `ollama`, `PyMuPDF`, and `python-docx`.

## Next chapter

Chapter 14 — Building Database-Connected AI Agents with Python.
