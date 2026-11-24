"""End-to-end API tests (needs fastapi, httpx, python-multipart)."""
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main

DOCKER = (b"Docker volumes keep persistent data when a container is removed. "
          b"A volume is mounted into the container with the -v flag.")


class APITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        env = patch.dict("os.environ", {
            "DATA_DIR": self.tmp.name, "EMBEDDINGS": "hash", "LLM_PROVIDER": "none",
        })
        env.start()
        self.addCleanup(env.stop)
        self.client = TestClient(main.app)
        self.client.__enter__()           # runs lifespan -> builds the RAG service

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.tmp.cleanup()

    def upload(self, name="docker.txt", data=DOCKER):
        return self.client.post("/api/documents", files={"file": (name, data, "text/plain")})

    def test_health(self):
        body = self.client.get("/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["llm"], "none")

    def test_upload_list_ask_delete(self):
        res = self.upload()
        self.assertEqual(res.status_code, 201)
        doc_id = res.json()["doc_id"]
        self.assertEqual(len(self.client.get("/api/documents").json()), 1)

        answer = self.client.post("/api/ask", json={"question": "How do docker volumes keep data?"}).json()
        self.assertEqual(answer["sources"][0]["filename"], "docker.txt")

        self.assertEqual(self.client.delete(f"/api/documents/{doc_id}").status_code, 200)
        self.assertEqual(self.client.delete(f"/api/documents/{doc_id}").status_code, 404)

    def test_ask_with_no_documents(self):
        body = self.client.post("/api/ask", json={"question": "anything at all"}).json()
        self.assertEqual(body["sources"], [])

    def test_bad_uploads(self):
        self.assertEqual(self.upload("malware.exe", b"x").status_code, 400)
        self.assertEqual(self.upload("empty.txt", b"  ").status_code, 400)

    def test_upload_too_large(self):
        with patch.object(main, "MAX_UPLOAD_BYTES", 10):
            self.assertEqual(self.upload(data=b"x" * 50).status_code, 413)

    def test_question_validation(self):
        self.assertEqual(self.client.post("/api/ask", json={"question": "hi"}).status_code, 422)
        self.assertEqual(self.client.post("/api/ask", json={"question": "valid question", "top_k": 99}).status_code, 422)

    def test_index_page(self):
        self.assertEqual(self.client.get("/").status_code, 200)


if __name__ == "__main__":
    unittest.main()
