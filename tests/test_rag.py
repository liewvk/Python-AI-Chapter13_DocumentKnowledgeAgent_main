"""Offline tests; no running Ollama server required."""
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

# Allow educational offline tests to run before the ollama package is installed.
try:
    import ollama  # noqa: F401
except ImportError:
    shim = ModuleType('ollama')
    shim.Client = object
    shim.ResponseError = type('ResponseError', (Exception,), {})
    sys.modules['ollama'] = shim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from document_loader import read_document, split_into_chunks, extract_chunks
from rag_service import KnowledgeRAG, cosine_similarity
from settings import NO_ANSWER


class FakeOllama:
    def __init__(self):
        self.chat_requests = []
        self.embedded_count = 0

    def embed(self, model, input):
        items = [input] if isinstance(input, str) else input
        vectors = []
        for item in items:
            words = item.lower()
            vector = [
                1 + words.count('refund') * 5 + words.count('tuition') * 2,
                1 + words.count('visitor') * 5 + words.count('reception') * 2,
                1 + words.count('library') * 5 + words.count('borrow') * 3,
                1 + words.count('password') * 6 + words.count('wi-fi') * 2,
            ]
            magnitude = math.sqrt(sum(v * v for v in vector))
            vectors.append([v / magnitude for v in vector])
        self.embedded_count += len(items)
        return SimpleNamespace(embeddings=vectors)

    def chat(self, **kwargs):
        self.chat_requests.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(
            content='Eligible prepaid tuition has a 75% refund [1].'))


class DocumentLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.knowledge = Path(__file__).resolve().parents[1] / 'knowledge'

    def test_pdf_extracts_page_locations(self):
        sections = read_document(self.knowledge / 'admissions_and_fees.pdf')
        self.assertEqual(sections[0]['location'], 'Page 1')
        self.assertEqual(sections[1]['location'], 'Page 2')
        self.assertIn('RM80', sections[1]['text'])
        self.assertIn('75 percent', ' '.join(sections[1]['text'].split()))

    def test_docx_extracts_paragraphs_and_table_in_order(self):
        sections = read_document(self.knowledge / 'staff_procedures.docx')
        text = ' '.join(section['text'] for section in sections)
        self.assertIn('visitor register', text)
        self.assertIn('Campus network outage', text)
        self.assertTrue(any(s['location'].startswith('Table ') for s in sections))

    def test_text_and_markdown_extract(self):
        for filename in ('library_faq.txt', 'it_knowledge_base.md'):
            self.assertEqual(read_document(self.knowledge / filename)[0]['location'], 'Text')

    def test_chunk_overlap_and_source_traceability(self):
        parts = split_into_chunks(' '.join(f'w{i}' for i in range(25)), size=12, overlap=4)
        self.assertEqual(parts[0].split()[-4:], parts[1].split()[:4])
        _, chunks = extract_chunks(self.knowledge)
        self.assertTrue(any(c['source'].endswith('.pdf') for c in chunks))
        self.assertTrue(all('location' in c and 'id' in c for c in chunks))


class RAGTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name) / 'knowledge'
        self.folder.mkdir()
        (self.folder / 'policy.txt').write_text(
            'Tuition refunds are available when a refund request is approved. '
            'The refund amount is 75 percent of prepaid tuition. '
            'The registration fee is non-refundable.', encoding='utf-8')
        self.client = FakeOllama()
        self.path = Path(self.temp.name) / 'index.json'
        self.agent = KnowledgeRAG(client=self.client, folder=self.folder, index_file=self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_cosine(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0)

    def test_index_persists_and_detects_changes(self):
        self.assertEqual(self.agent.build_index()[0], 1)
        self.assertTrue(self.agent.is_current())
        reloaded = KnowledgeRAG(client=self.client, folder=self.folder, index_file=self.path)
        self.assertEqual(len(reloaded.records), len(self.agent.records))
        (self.folder / 'policy.txt').write_text('The policy was changed.', encoding='utf-8')
        self.assertFalse(reloaded.is_current())
        with self.assertRaisesRegex(ValueError, 'Documents changed'):
            reloaded.search('refund')

    def test_query_has_sources_and_chat_receives_excerpts(self):
        self.agent.build_index()
        matches = self.agent.search('What tuition refund is available?')
        self.assertTrue(matches)
        self.assertEqual(matches[0]['source'], 'policy.txt')
        answer, references = self.agent.ask('What tuition refund is available?')
        self.assertIn('75%', answer)
        self.assertTrue(references)
        prompt = self.client.chat_requests[0]['messages'][1]['content']
        self.assertIn('Document: policy.txt', prompt)
        self.assertIn('SOURCE EXCERPTS', prompt)

    def test_no_index_shows_helpful_error(self):
        with self.assertRaisesRegex(ValueError, 'No document index'):
            self.agent.search('refund')


if __name__ == '__main__':
    unittest.main()
