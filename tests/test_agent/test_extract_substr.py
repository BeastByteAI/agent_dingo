import unittest
from agent_dingo.agent.docgen import extract_substr


class TestExtractSubstr(unittest.TestCase):
    def test_extract_substr(self):
        input_string = """Extracts the desription and args from a docstring.

        Args:
            input_string (str): The docstring to extract the description and args from.

        Returns:
            str: Reduced docstring containing only the description and the args.
        """
        expected_output = """Extracts the desription and args from a docstring.

        Args:
            input_string (str): The docstring to extract the description and args from."""
        self.assertEqual(extract_substr(input_string), expected_output)
