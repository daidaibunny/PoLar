import tempfile
import unittest
from collections import Counter
from pathlib import Path

from reconstruction.data import (
    DataIntegrityError,
    deduplicate_query_records,
    sha256_file,
    split_level_records,
)


class DeduplicateQueryRecordsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.query_info_rows = [
            {
                "query_id": "q1",
                "level": 1,
                "domain": "Algebra",
                "pass_rate": 0.5,
            },
            {
                "query_id": "q2",
                "level": 1,
                "domain": "Algebra",
                "pass_rate": 0.25,
            },
            {
                "query_id": "q3",
                "level": 1,
                "domain": "Geometry",
                "pass_rate": 0.75,
            },
        ]

    def test_deduplicates_response_rows_by_query_id(self) -> None:
        pool_rows = [
            {"query_id": "q1", "query": "One?", "gt_ans": "1"},
            {"query_id": "q1", "query": "One?", "gt_ans": "1"},
            {"query_id": "q2", "query": "Two?", "gt_ans": "2"},
            {"query_id": "q3", "query": "Three?", "gt_ans": "3"},
            {"query_id": "unknown", "query": "Unknown?", "gt_ans": "0"},
        ]

        result = deduplicate_query_records(self.query_info_rows, pool_rows)

        self.assertEqual([row["query_id"] for row in result.records], ["q1", "q2", "q3"])
        self.assertEqual(result.records[0]["dart_pass_rate"], 0.5)
        self.assertEqual(result.stats.duplicate_response_rows, 1)
        self.assertEqual(result.stats.unknown_query_rows, 1)
        self.assertEqual(result.stats.missing_query_ids, ())

    def test_rejects_conflicting_question_or_answer(self) -> None:
        pool_rows = [
            {"query_id": "q1", "query": "One?", "gt_ans": "1"},
            {"query_id": "q1", "query": "Different?", "gt_ans": "1"},
        ]

        with self.assertRaises(DataIntegrityError):
            deduplicate_query_records(self.query_info_rows, pool_rows)

    def test_reports_missing_answers_and_queries(self) -> None:
        pool_rows = [
            {"query_id": "q1", "query": "One?", "gt_ans": ""},
            {"query_id": "q2", "query": "Two?", "gt_ans": "2"},
        ]

        result = deduplicate_query_records(self.query_info_rows, pool_rows)

        self.assertEqual(result.stats.missing_answers, 1)
        self.assertEqual(result.stats.missing_query_ids, ("q3",))


class SplitLevelRecordsTest(unittest.TestCase):
    @staticmethod
    def _records(count: int) -> list[dict]:
        return [
            {
                "query_id": f"q{index:04d}",
                "question": f"Question {index}",
                "gt_ans": str(index),
                "level": 1,
                "domain": "Algebra" if index % 3 else "Geometry",
                "dart_pass_rate": index / max(1, count),
            }
            for index in range(count)
        ]

    def test_uses_all_records_when_level_has_fewer_than_2000(self) -> None:
        split = split_level_records(self._records(17), seed=42)

        self.assertEqual(len(split.train), 11)
        self.assertEqual(len(split.validation), 2)
        self.assertEqual(len(split.test), 4)
        self.assertEqual(split.available_count, 17)
        self.assertEqual(split.selected_count, 17)

    def test_caps_large_level_at_exact_paper_split_size(self) -> None:
        split = split_level_records(self._records(2100), seed=42)

        self.assertEqual(len(split.train), 1250)
        self.assertEqual(len(split.validation), 250)
        self.assertEqual(len(split.test), 500)
        self.assertEqual(split.available_count, 2100)
        self.assertEqual(split.selected_count, 2000)
        selected = split.train + split.validation + split.test
        self.assertEqual(
            Counter(row["domain"] for row in selected),
            Counter({"Algebra": 1333, "Geometry": 667}),
        )

    def test_split_is_deterministic_and_has_no_query_overlap(self) -> None:
        first = split_level_records(self._records(101), seed=42)
        second = split_level_records(self._records(101), seed=42)

        self.assertEqual(first, second)
        train_ids = {row["query_id"] for row in first.train}
        validation_ids = {row["query_id"] for row in first.validation}
        test_ids = {row["query_id"] for row in first.test}
        self.assertFalse(train_ids & validation_ids)
        self.assertFalse(train_ids & test_ids)
        self.assertFalse(validation_ids & test_ids)


class HashTest(unittest.TestCase):
    def test_sha256_file_uses_file_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_bytes(b"PoLar\n")

            self.assertEqual(
                sha256_file(path),
                "061738f85e5d84a95c628981b28d77573fc65ac72e60b2aa39ec80c04ed87d52",
            )


if __name__ == "__main__":
    unittest.main()
