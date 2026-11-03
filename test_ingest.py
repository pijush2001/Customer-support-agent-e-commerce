"""Real PDF and embedding integration checks. First run needs the model download."""
import tempfile
import unittest
from pathlib import Path

from reportlab.pdfgen import canvas

from ingest import (CHUNK_SIZE, generate_sample_pdf, ingest_documents,
                    load_pdf_chunks, open_collection, search_policies)


class PolicyIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls.temp.name)
        cls.source = cls.root / "policies"
        cls.database = cls.root / "chroma"
        generate_sample_pdf(cls.source)
        cls.count = ingest_documents(cls.source, cls.database)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_chunks_have_text_and_provenance(self):
        chunks = load_pdf_chunks(self.source)
        self.assertGreater(len(chunks), 6)
        self.assertEqual(len(chunks), len({chunk["id"] for chunk in chunks}))
        self.assertEqual({chunk["metadata"]["page"] for chunk in chunks}, set(range(1, 7)))
        for chunk in chunks:
            self.assertTrue(chunk["text"].strip())
            self.assertLessEqual(len(chunk["text"]), CHUNK_SIZE)
            self.assertEqual(chunk["metadata"]["source"], "fictional_store_policies.pdf")

    def test_semantic_queries_find_correct_policy(self):
        cases = [("Can I send my purchase back?", "page-1", "30 days"),
                 ("When will I get my money back?", "page-2", "5 business days"),
                 ("Can I stop an order before dispatch?", "page-3", "human specialist")]
        for question, page, expected in cases:
            with self.subTest(question=question):
                matches = search_policies(question, k=1, db_dir=self.database)
                self.assertEqual(len(matches), 1)
                self.assertIn(page, matches[0][0])
                self.assertIn(expected, matches[0][1])

    def test_unrelated_queries_and_empty_queries_abstain(self):
        for question in ("What is the meaning of life?", "What is the capital of France?", ""):
            self.assertEqual(search_policies(question, db_dir=self.database), [])
        self.assertEqual(search_policies("return policy", k=0, db_dir=self.database), [])

    def test_vectors_persist_and_reingestion_does_not_duplicate(self):
        self.assertEqual(ingest_documents(self.source, self.database), self.count)
        collection = open_collection(self.database)
        result = collection.get(limit=1, include=["embeddings"])
        self.assertEqual(len(result["embeddings"][0]), 384)
        self.assertEqual(collection.count(), self.count)

    def test_updated_and_deleted_pdfs_remove_stale_chunks(self):
        source = self.root / "update_policies"
        database = self.root / "update_chroma"
        generate_sample_pdf(source)
        ingest_documents(source, database)
        path = source / "fictional_store_policies.pdf"
        pdf = canvas.Canvas(str(path))
        pdf.drawString(40, 750, "Updated return policy: returns are accepted within 45 days.")
        pdf.save()
        self.assertEqual(ingest_documents(source, database), 1)
        self.assertIn("45 days", open_collection(database).get()["documents"][0])
        extra = source / "extra.pdf"
        pdf = canvas.Canvas(str(extra))
        pdf.drawString(40, 750, "Express shipping takes 2 business days.")
        pdf.save()
        self.assertEqual(ingest_documents(source, database), 2)
        extra.unlink()
        self.assertEqual(ingest_documents(source, database), 1)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "No PDFs"):
            ingest_documents(source, database)
        self.assertEqual(open_collection(database).count(), 1)

    def test_blank_pdf_requires_ocr(self):
        source = self.root / "blank_policies"
        source.mkdir()
        pdf = canvas.Canvas(str(source / "blank.pdf"))
        pdf.showPage()
        pdf.save()
        with self.assertRaisesRegex(ValueError, "require OCR"):
            load_pdf_chunks(source)

    def test_missing_index_gives_setup_instruction(self):
        with self.assertRaisesRegex(RuntimeError, "python ingest.py --sample"):
            search_policies("return policy", db_dir=self.root / "missing_chroma")


if __name__ == "__main__":
    unittest.main()