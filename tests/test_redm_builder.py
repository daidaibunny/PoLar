import json
import tempfile
import unittest
from pathlib import Path

from reconstruction.redm_builder import (
	DatasetSource,
	SourceFile,
	build_redm_human,
	build_redm_public,
)
from reconstruction.data import sha256_file


class ReDMHumanBuilderTest(unittest.TestCase):
	def test_writes_five_disjoint_level_files_and_a_hash_manifest(self) -> None:
		query_info_rows = [
			{
				"query_id": f"q{level}-{index}",
				"level": level,
				"domain": "Algebra" if index % 2 else "Geometry",
				"pass_rate": index / 10,
			}
			for level in range(1, 6)
			for index in range(8)
		]
		pool_rows = [
			{
				"query_id": row["query_id"],
				"query": f"Question {row['query_id']}",
				"gt_ans": "1",
			}
			for row in query_info_rows
		]
		source = DatasetSource(
			repository="example/source",
			revision="a" * 40,
			files=(SourceFile("data.parquet", "b" * 64, 100),),
		)

		with tempfile.TemporaryDirectory() as directory:
			result = build_redm_human(
				query_info_rows,
				pool_rows,
				Path(directory),
				(source,),
			)
			manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

			self.assertEqual(result.unique_questions, 40)
			self.assertEqual(manifest["statistics"]["by_level"]["1"], 8)
			self.assertEqual(manifest["sources"][0]["revision"], "a" * 40)
			for level in range(1, 6):
				level_path = Path(directory) / f"diff-{level}.json"
				payload = json.loads(level_path.read_text(encoding="utf-8"))
				all_ids = [
					row["query_id"]
					for split_name in ("train", "validation", "test")
					for row in payload[split_name]
				]
				self.assertEqual(len(all_ids), 8)
				self.assertEqual(len(all_ids), len(set(all_ids)))
				self.assertEqual(
					len(manifest["levels"][str(level)]["output_sha256"]),
					64,
				)
				self.assertEqual(
					manifest["levels"][str(level)]["output_sha256"],
					sha256_file(level_path),
				)

	def test_refuses_to_overwrite_an_existing_output(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			output_path = Path(directory) / "diff-1.json"
			output_path.write_text("existing", encoding="utf-8")

			with self.assertRaises(FileExistsError):
				build_redm_human([], [], Path(directory), ())


class ReDMPublicBuilderTest(unittest.TestCase):
	def test_writes_five_equal_pass_rate_difficulty_files(self) -> None:
		query_info_rows = [
			{
				"query_id": f"q{index:02d}",
				"level": index % 5 + 1,
				"domain": "Algebra" if index % 2 else "Geometry",
				"pass_rate": 1.0 - index / 50,
			}
			for index in range(50)
		]
		pool_rows = [
			{
				"query_id": row["query_id"],
				"query": f"Question {row['query_id']}",
				"gt_ans": "1",
			}
			for row in query_info_rows
		]
		source = DatasetSource(
			repository="example/source",
			revision="a" * 40,
			files=(SourceFile("data.parquet", "b" * 64, 100),),
		)

		with tempfile.TemporaryDirectory() as directory:
			result = build_redm_public(
				query_info_rows,
				pool_rows,
				Path(directory),
				(source,),
			)
			manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

			self.assertEqual(result.unique_questions, 50)
			self.assertEqual(manifest["dataset_name"], "ReDM-Public")
			self.assertEqual(manifest["statistics"]["by_public_difficulty"], {
				str(level): 10 for level in range(1, 6)
			})
			all_ids = []
			for level in range(1, 6):
				level_path = Path(directory) / f"diff-{level}.json"
				payload = json.loads(level_path.read_text(encoding="utf-8"))
				self.assertEqual(payload["difficulty_level"], level)
				self.assertEqual(payload["split_counts"], {
					"train": 6,
					"validation": 1,
					"test": 3,
				})
				all_ids.extend(
					row["query_id"]
					for split_name in ("train", "validation", "test")
					for row in payload[split_name]
				)
			self.assertEqual(len(all_ids), 50)
			self.assertEqual(len(all_ids), len(set(all_ids)))


if __name__ == "__main__":
	unittest.main()
