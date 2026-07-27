import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from reconstruction.data import DataIntegrityError, sha256_file
from reconstruction.redm_sources import (
	load_math_train_archive,
	validate_math_archive,
)


class MathArchiveSourceTest(unittest.TestCase):
	def test_preserves_query_paths_and_extracts_boxed_answers(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			archive_path = Path(directory) / "MATH.zip"
			payload = {
				"problem": "What is one plus one?",
				"solution": "Therefore the result is \\boxed{2}.",
				"level": "Level 1",
				"type": "Algebra",
			}
			with zipfile.ZipFile(archive_path, "w") as archive:
				archive.writestr("MATH/train/algebra/1.json", json.dumps(payload))

			records = load_math_train_archive(archive_path)

			self.assertEqual(len(records), 1)
			self.assertEqual(records[0]["query_id"], "MATH/train/algebra/1.json")
			self.assertEqual(records[0]["query"], "What is one plus one?")
			self.assertEqual(records[0]["gt_ans"], "2")
			self.assertEqual(records[0]["level"], 1)
			self.assertEqual(records[0]["domain"], "Algebra")

	def test_rejects_missing_boxed_answer(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			archive_path = Path(directory) / "MATH.zip"
			payload = {
				"problem": "Question",
				"solution": "No boxed final answer.",
				"level": "Level 1",
				"type": "Algebra",
			}
			with zipfile.ZipFile(archive_path, "w") as archive:
				archive.writestr("MATH/train/algebra/1.json", json.dumps(payload))

			with self.assertRaises(DataIntegrityError):
				load_math_train_archive(archive_path)

	def test_uses_only_verified_dart_pool_answer_corrections(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			archive_path = Path(directory) / "MATH.zip"
			payload = {
				"problem": "How many primes are present?",
				"solution": "Therefore there are \\boxed{} primes.",
				"level": "Level 4",
				"type": "Number Theory",
			}
			with zipfile.ZipFile(archive_path, "w") as archive:
				archive.writestr(
					"MATH/train/number_theory/7115.json",
					json.dumps(payload),
				)

			records = load_math_train_archive(archive_path)

			self.assertEqual(records[0]["gt_ans"], "0")
			self.assertEqual(records[0]["answer_source"], "DART-Math response pool")

	def test_validates_exact_archive_sha256(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			archive_path = Path(directory) / "MATH.tar"
			archive_path.write_bytes(b"math archive")
			expected = sha256_file(archive_path)

			validate_math_archive(archive_path, expected)
			with self.assertRaises(DataIntegrityError):
				validate_math_archive(archive_path, "0" * 64)


if __name__ == "__main__":
	unittest.main()
