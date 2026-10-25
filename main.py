"""Chapter 13: PDF, DOCX, TXT, and Markdown knowledge base agent."""
import sys

from ollama import ResponseError

from document_loader import list_documents, read_document
from rag_service import KnowledgeRAG
from settings import (
    CHAT_MODEL, EMBEDDING_MODEL, KNOWLEDGE_DIR, MAX_QUESTION_LENGTH,
)


def show_help():
    print("\nCommands:")
    print("  /index                  Build or rebuild the knowledge index")
    print("  /docs                   List supported files in knowledge/")
    print("  /preview filename       Show extracted document text (first 1000 chars)")
    print("  /search your question   Show matching passages and similarity scores")
    print("  /status                 Show index information")
    print("  /help                   Show the available commands")
    print("  /exit                   Close the application")
    print("  Or type a normal question to ask the knowledge base.")


def preview(name):
    paths = {p.name.lower(): p for p in list_documents()}
    path = paths.get(name.lower())
    if path is None:
        raise ValueError("Document not found; use /docs for exact filenames.")
    sections = read_document(path)
    result = "\n\n".join(f"[{s['location']}]\n{s['text']}" for s in sections)
    print("\n", result[:1000], sep="")
    if len(result) > 1000:
        print("... [preview truncated]")


def main():
    print("CHAPTER 13 - MULTI-FORMAT DOCUMENT KNOWLEDGE AGENT")
    print("Chat model:", CHAT_MODEL)
    print("Embedding model:", EMBEDDING_MODEL)
    print("Python interpreter:", sys.executable)
    print("Knowledge folder:", KNOWLEDGE_DIR)
    try:
        agent = KnowledgeRAG()
    except (ValueError, OSError) as exc:
        print("Cannot load index:", exc)
        print("Move the invalid data/document_index.json and restart.")
        return

    show_help()
    if not agent.records:
        print("\nNo index found. Type /index to process the sample documents.")
    elif not agent.is_current():
        print("\nDocuments changed. Type /index to rebuild the index.")

    while True:
        try:
            line = input("\nYou: ").strip()
            if not line:
                continue
            action, _, argument = line.partition(" ")
            action = action.lower()
            argument = argument.strip()

            if action == "/exit":
                print("Goodbye!")
                break
            elif action == "/help":
                show_help()
            elif action == "/index":
                print("Extracting documents and generating embeddings...")
                documents, chunks = agent.build_index()
                print(f"Index ready: {documents} documents, {chunks} chunks.")
            elif action == "/docs":
                for path in list_documents():
                    print(" -", path.name)
            elif action == "/preview":
                if not argument:
                    print("Usage: /preview filename.pdf")
                else:
                    preview(argument)
            elif action == "/status":
                print("Indexed documents:", len(agent.sources()))
                print("Indexed chunks:", len(agent.records))
                print("Index current:", agent.is_current())
                print("Embedding model:", EMBEDDING_MODEL)
            elif action == "/search":
                if not argument:
                    print("Usage: /search your question")
                else:
                    results = agent.search(argument)
                    if not results:
                        print("No passages met the similarity threshold.")
                    for result in results:
                        print(
                            f"\n[{result['id']}] {result['source']} | "
                            f"{result['location']} | similarity={result['score']:.3f}"
                        )
                        print(result["text"][:450])
            elif action.startswith("/"):
                print("Unknown command; type /help.")
            elif len(line) > MAX_QUESTION_LENGTH:
                print("Question exceeds the length limit.")
            else:
                answer, evidence = agent.ask(line)
                print("\nAssistant:", answer)
                if evidence:
                    print("\nRetrieved source passages:")
                    for number, match in enumerate(evidence, 1):
                        print(f"[{number}] {match['source']} - {match['location']}")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        except (ResponseError, ConnectionError, TimeoutError,
                ValueError, OSError, RuntimeError) as exc:
            print("Operation failed:", exc)
            if "not found" in str(exc).lower() and "model" in str(exc).lower():
                print("Install the models with ollama pull commands (see README).")
        except Exception as exc:
            print("Unexpected error:", exc)


if __name__ == "__main__":
    main()
