"""Tests for the framework-independent pipeline (needs numpy + pypdf only)."""
import tempfile
import unittest

import numpy as np

from app.chunking import chunk_text
from app.embeddings import HashingEmbedder, tokenize
from app.llm import build_user_prompt
from app.loaders import extract_text
from app.rag import NOT_FOUND, RAGService
from app.vectorstore import VectorStore

DOCKER = ("Docker volumes keep persistent data when a container is removed. "
          "A volume is mounted into the container with the -v flag.")
GIT = ("Git branches let you work on features separately. "
       "Use git merge to combine two branches and resolve any conflict.")


def make_pdf(text: str) -> bytes:
    """Build a minimal one-page PDF containing `text` (no parentheses in text)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{obj}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


class ChunkingTests(unittest.TestCase):
    def test_overlap_and_coverage(self):
        words = [f"w{i}" for i in range(100)]
        chunks = chunk_text(" ".join(words), chunk_size=40, overlap=10)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].split()[-10:], chunks[1].split()[:10])   # overlap is shared
        self.assertEqual(chunks[-1].split()[-1], "w99")                      # nothing is dropped

    def test_short_text_is_one_chunk(self):
        self.assertEqual(chunk_text("just a few words", 120, 30), ["just a few words"])

    def test_empty_and_bad_params(self):
        self.assertEqual(chunk_text("   ", 10, 2), [])
        with self.assertRaises(ValueError):
            chunk_text("abc", 10, 10)


class LoaderTests(unittest.TestCase):
    def test_text_cleanup(self):
        self.assertEqual(extract_text("a.txt", b"hello    world\n\n\n\nbye"), "hello world\n\nbye")

    def test_pdf_extraction(self):
        text = extract_text("doc.pdf", make_pdf("Volumes keep data safe"))
        self.assertIn("Volumes keep data safe", text)

    def test_rejects_bad_files(self):
        with self.assertRaises(ValueError):
            extract_text("virus.exe", b"x")
        with self.assertRaises(ValueError):
            extract_text("broken.pdf", b"this is not a pdf")


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.emb = HashingEmbedder()

    def test_tokenize_drops_stopwords_and_plurals(self):
        self.assertEqual(tokenize("What are the Docker volumes?"), ["docker", "volume"])

    def test_vectors_are_unit_length_and_deterministic(self):
        a = self.emb.embed(["docker volumes persist data"])
        b = self.emb.embed(["docker volumes persist data"])
        self.assertTrue(np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5))
        self.assertTrue(np.array_equal(a, b))

    def test_related_text_scores_higher(self):
        q, near, far = self.emb.embed(["keep data in docker volumes", DOCKER, GIT])
        self.assertGreater(float(q @ near), float(q @ far))


class VectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.emb = HashingEmbedder()

    def tearDown(self):
        self.tmp.cleanup()

    def test_add_search_delete(self):
        store = VectorStore(self.tmp.name, self.emb.name)
        d1 = store.add_document("docker.txt", [DOCKER], self.emb.embed([DOCKER]))
        store.add_document("git.txt", [GIT], self.emb.embed([GIT]))
        hits = store.search(self.emb.embed(["docker volume data"])[0], k=2)
        self.assertEqual(hits[0]["filename"], "docker.txt")
        self.assertTrue(store.delete_document(d1))
        self.assertFalse(store.delete_document(d1))
        self.assertEqual([h["filename"] for h in store.search(self.emb.embed(["docker"])[0], 5)], ["git.txt"])

    def test_persistence_across_restarts(self):
        store = VectorStore(self.tmp.name, self.emb.name)
        store.add_document("docker.txt", [DOCKER], self.emb.embed([DOCKER]))
        reloaded = VectorStore(self.tmp.name, self.emb.name)
        self.assertEqual(len(reloaded.list_documents()), 1)
        self.assertEqual(reloaded.search(self.emb.embed(["docker volume"])[0], 1)[0]["filename"], "docker.txt")

    def test_embedder_mismatch_is_rejected(self):
        VectorStore(self.tmp.name, self.emb.name).add_document("a.txt", [DOCKER], self.emb.embed([DOCKER]))
        with self.assertRaises(ValueError):
            VectorStore(self.tmp.name, "some-other-embedder")

    def test_empty_store_search(self):
        self.assertEqual(VectorStore(self.tmp.name, self.emb.name).search(self.emb.embed(["x"])[0]), [])


class FakeLLM:
    name = "fake"

    def __init__(self):
        self.last_user = None

    def generate(self, system, user):
        self.last_user = user
        return "Answer [1]"


class RAGTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        emb = HashingEmbedder()
        self.llm = FakeLLM()
        self.rag = RAGService(VectorStore(self.tmp.name, emb.name), emb, self.llm)
        self.rag.ingest("docker.txt", DOCKER.encode())
        self.rag.ingest("git.txt", GIT.encode())

    def tearDown(self):
        self.tmp.cleanup()

    def test_ask_retrieves_right_source_and_builds_prompt(self):
        result = self.rag.ask("How do I keep data in a docker volume?")
        self.assertEqual(result["answer"], "Answer [1]")
        self.assertEqual(result["sources"][0]["filename"], "docker.txt")
        self.assertIn("[1] (source: docker.txt)", self.llm.last_user)
        self.assertIn("Question: How do I keep data", self.llm.last_user)

    def test_irrelevant_question_skips_llm(self):
        result = self.rag.ask("zebra quantum xylophone")
        self.assertEqual(result["answer"], NOT_FOUND)
        self.assertEqual(result["sources"], [])
        self.assertIsNone(self.llm.last_user)

    def test_extractive_mode_without_llm(self):
        self.rag.llm = None
        result = self.rag.ask("combine branches with git merge")
        self.assertEqual(result["provider"], "none")
        self.assertIn("git.txt", result["answer"])

    def test_empty_file_rejected(self):
        with self.assertRaises(ValueError):
            self.rag.ingest("empty.txt", b"   \n  ")

    def test_prompt_numbering(self):
        prompt = build_user_prompt("Q?", [{"filename": "a", "text": "x"}, {"filename": "b", "text": "y"}])
        self.assertIn("[2] (source: b)", prompt)


if __name__ == "__main__":
    unittest.main()
