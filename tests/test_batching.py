import unittest

from reconstruction.batching import BatchMeasurement, choose_largest_suitable_batch


class BatchSelectionTest(unittest.TestCase):
	def test_selects_largest_near_peak_batch_with_memory_headroom(self) -> None:
		measurements = (
			BatchMeasurement(8, True, 10.0, 20),
			BatchMeasurement(16, True, 18.0, 30),
			BatchMeasurement(32, True, 20.0, 55),
			BatchMeasurement(64, True, 19.2, 73),
			BatchMeasurement(128, True, 21.0, 75),
		)

		selected = choose_largest_suitable_batch(
			measurements=measurements,
			total_memory_bytes=80,
			minimum_peak_throughput_fraction=0.95,
			memory_headroom_fraction=0.10,
		)

		self.assertEqual(selected, 32)

	def test_rejects_all_failed_or_memory_exhausting_measurements(self) -> None:
		with self.assertRaises(RuntimeError):
			choose_largest_suitable_batch(
				measurements=(BatchMeasurement(8, False, 0.0, 0),),
				total_memory_bytes=80,
			)
		with self.assertRaises(RuntimeError):
			choose_largest_suitable_batch(
				measurements=(BatchMeasurement(8, True, 10.0, 79),),
				total_memory_bytes=80,
				memory_headroom_fraction=0.10,
			)


if __name__ == "__main__":
	unittest.main()
