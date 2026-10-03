import json
import unittest
from pathlib import Path

from rag.ingest import extract_file


class StructuredUploadsTest(unittest.TestCase):
    def test_extract_file_supports_csv_and_json(self):
        tmp_dir = Path("tmp_structured_uploads")
        tmp_dir.mkdir(exist_ok=True)

        try:
            csv_path = tmp_dir / "sample.csv"
            csv_path.write_text("name,score\nAlice,95\nBob,88\n", encoding="utf-8")

            csv_result = extract_file(csv_path)
            self.assertTrue(csv_result)
            self.assertTrue(any("Alice" in item["text"] for item in csv_result))

            json_path = tmp_dir / "sample.json"
            json_path.write_text(
                json.dumps({"title": "RAG", "items": [{"name": "Alice"}]}),
                encoding="utf-8",
            )

            json_result = extract_file(json_path)
            self.assertTrue(json_result)
            self.assertTrue(any("RAG" in item["text"] for item in json_result))
        finally:
            for path in sorted(tmp_dir.iterdir(), reverse=True):
                if path.is_file():
                    path.unlink()
            tmp_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
