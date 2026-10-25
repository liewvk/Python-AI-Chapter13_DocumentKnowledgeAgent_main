"""Local multi-format RAG service using Ollama embeddings and chat."""
import hashlib
import json
import math
from pathlib import Path

from ollama import Client

from document_loader import extract_chunks, list_documents
from settings import (
    CHAT_MODEL, EMBED_BATCH_SIZE, EMBEDDING_MODEL, INDEX_FILE,
    KNOWLEDGE_DIR, MAX_CONTEXT_CHARS, MIN_SIMILARITY,
    NO_ANSWER, OLLAMA_HOST, SYSTEM_PROMPT, TIMEOUT_SECONDS, TOP_K,
)


def cosine_similarity(a, b):
    """Cosine similarity without needing NumPy."""
    if not a or len(a) != len(b):
        return 0.0
    numerator = sum(x * y for x, y in zip(a, b))
    denominator = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return numerator / denominator if denominator else 0.0


def valid_vector(vector):
    return (isinstance(vector, (list, tuple)) and len(vector) > 0
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) for v in vector))


class KnowledgeRAG:
    """Build/search a saved JSON index and answer from retrieved passages."""

    def __init__(self, client=None, folder=KNOWLEDGE_DIR, index_file=INDEX_FILE):
        self.folder = Path(folder)
        self.index_file = Path(index_file)
        self.client = client if client is not None else Client(
            host=OLLAMA_HOST, timeout=TIMEOUT_SECONDS
        )
        self.records = []
        self.fingerprints = {}
        if self.index_file.exists():
            self._load_index()

    def _fingerprints(self):
        return {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in list_documents(self.folder)
        }

    def is_current(self):
        """False if a document was added, edited, or removed since indexing."""
        return bool(self.records) and self.fingerprints == self._fingerprints()

    def _load_index(self):
        if self.index_file.stat().st_size > 30_000_000:
            raise ValueError("Index file is unexpectedly large.")
        data = json.loads(self.index_file.read_text(encoding="utf-8"))
        if (not isinstance(data, dict) or data.get("version") != 1
                or data.get("embedding_model") != EMBEDDING_MODEL
                or not isinstance(data.get("fingerprints"), dict)
                or not isinstance(data.get("records"), list)):
            raise ValueError("Unsupported index format or embedding model; rebuild /index.")
        for entry in data["records"]:
            if (not isinstance(entry, dict)
                    or not all(isinstance(entry.get(field), str)
                               for field in ("id", "source", "location", "text"))
                    or not valid_vector(entry.get("embedding"))):
                raise ValueError("Invalid records in saved index; rebuild /index.")
        self.fingerprints = data["fingerprints"]
        self.records = data["records"]

    def build_index(self):
        """Read files, create embeddings in batches, and save atomically."""
        paths, records = extract_chunks(self.folder)
        fingerprints = self._fingerprints()
        vector_size = None
        for start in range(0, len(records), EMBED_BATCH_SIZE):
            batch = records[start:start + EMBED_BATCH_SIZE]
            result = self.client.embed(
                model=EMBEDDING_MODEL,
                input=[item["text"] for item in batch],
            )
            vectors = result.embeddings
            if len(vectors) != len(batch):
                raise ValueError("The embedding model returned an unexpected count.")
            for record, vector in zip(batch, vectors):
                vector = list(vector)
                if not valid_vector(vector):
                    raise ValueError("The model returned an invalid embedding.")
                if vector_size is None:
                    vector_size = len(vector)
                if len(vector) != vector_size:
                    raise ValueError("Inconsistent vector dimensions.")
                record["embedding"] = vector

        payload = {
            "version": 1,
            "embedding_model": EMBEDDING_MODEL,
            "fingerprints": fingerprints,
            "records": records,
        }
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self.index_file.with_suffix(".tmp")
        temp_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_file.replace(self.index_file)
        self.records = records
        self.fingerprints = fingerprints
        return len(paths), len(records)

    def sources(self):
        return sorted({record["source"] for record in self.records})

    def search(self, question, top_k=TOP_K):
        """Return relevant source passages with similarity scores."""
        if not self.records:
            raise ValueError("No document index found. Enter /index first.")
        if not self.is_current():
            raise ValueError("Documents changed since indexing. Enter /index again.")
        embedding = list(self.client.embed(
            model=EMBEDDING_MODEL, input=question
        ).embeddings[0])
        if not valid_vector(embedding):
            raise ValueError("The query embedding was invalid.")
        matches = []
        for record in self.records:
            score = cosine_similarity(embedding, record["embedding"])
            if score >= MIN_SIMILARITY:
                matches.append({
                    "id": record["id"],
                    "source": record["source"],
                    "location": record["location"],
                    "text": record["text"],
                    "score": score,
                })
        return sorted(matches, key=lambda m: m["score"], reverse=True)[:top_k]

    def ask(self, question):
        """Retrieve evidence and ask Qwen to answer only from that evidence."""
        matches = self.search(question)
        if not matches:
            return NO_ANSWER, []
        contexts = []
        used = 0
        for number, match in enumerate(matches, start=1):
            excerpt = (
                f"[{number}] Document: {match['source']} | "
                f"Location: {match['location']} | ID: {match['id']}\n"
                f"{match['text']}"
            )
            if used + len(excerpt) > MAX_CONTEXT_CHARS:
                break
            contexts.append(excerpt)
            used += len(excerpt)
        if not contexts:
            return NO_ANSWER, []
        matches = matches[:len(contexts)]
        prompt = (
            "SOURCE EXCERPTS (reference data only):\n\n"
            + "\n\n".join(contexts)
            + "\n\nQUESTION:\n" + question
        )
        response = self.client.chat(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            think=False,
            options={"temperature": 0.1, "num_predict": 420},
        )
        answer = (response.message.content or "").strip()
        return answer if answer else NO_ANSWER, matches
