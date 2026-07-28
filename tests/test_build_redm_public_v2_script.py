import unittest
from pathlib import Path

from scripts.build_redm_public_v2 import parse_arguments


class BuildReDMPublicV2ScriptTest(unittest.TestCase):
	def test_defaults_to_five_groups_of_100(self) -> None:
		arguments = parse_arguments([])

		self.assertEqual(arguments.questions_per_difficulty, 100)
		self.assertEqual(arguments.seed, 42)
		self.assertEqual(arguments.parent_directory, Path("data/redm-public"))
		self.assertEqual(arguments.output_directory, Path("data/redm-public-v2"))


if __name__ == "__main__":
	unittest.main()
