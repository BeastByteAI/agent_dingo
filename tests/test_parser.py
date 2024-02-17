import unittest
from agent_dingo.agent.parser import parse


class TestParser(unittest.TestCase):
    def test_parse_google(self):
        docstring = """Parses a docstring.

        Args:
            docstring (str): The docstring to parse.

        Returns:
            dict: A dictionary containing the description and the arguments of the function.
        """
        expected_output = (
            {
                "description": "Parses a docstring.",
                "properties": {
                    "docstring": {
                        "type": "string",
                        "description": "The docstring to parse.",
                    }
                },
            },
            False,
        )
        self.assertEqual(parse(docstring), expected_output)

    def test_parse_numpy(self):
        docstring = """Parses a docstring.

        Parameters
        ----------
        docstring : str
            The docstring to parse.

        Returns
        -------
        dict
            A dictionary containing the description and the arguments of the function.
        """
        expected_output = (
            {
                "description": "Parses a docstring.",
                "properties": {
                    "docstring": {
                        "type": "string",
                        "description": "The docstring to parse.",
                    }
                },
            },
            False,
        )
        self.assertEqual(parse(docstring), expected_output)

    def test_parse_with_enum(self):
        docstring = """Parses a docstring.

        Parameters
        ----------
        arg1 : str
            The first argument.
        arg2 : int
            The second argument.
        arg3 : float
            The third argument.
        arg4 : bool
            The fourth argument.
        arg5 : list
            The fifth argument.
        arg6 : dict
            The sixth argument.
        arg7 : str
            The seventh argument. Enum: ['value1', 'value2', 'value3']

        Returns
        -------
        dict
            A dictionary containing the description and the arguments of the function.
        """
        expected_output = (
            {
                "description": "Parses a docstring.",
                "properties": {
                    "arg1": {"type": "string", "description": "The first argument."},
                    "arg2": {"type": "integer", "description": "The second argument."},
                    "arg3": {"type": "number", "description": "The third argument."},
                    "arg4": {"type": "boolean", "description": "The fourth argument."},
                    "arg5": {"type": "array", "description": "The fifth argument."},
                    "arg6": {"type": "object", "description": "The sixth argument."},
                    "arg7": {
                        "type": "string",
                        "description": "The seventh argument.",
                        "enum": ["value1", "value2", "value3"],
                    },
                },
            },
            False,
        )
        self.assertEqual(parse(docstring), expected_output)

    def test_parse_with_context(self):
        docstring = """Parses a docstring.

        Parameters
        ----------
        arg1 : str
            The first argument.
        chat_context : ChatContext
            The chat context.

        Returns
        -------
        dict
            A dictionary containing the description and the arguments of the function.
        """
        expected_output = (
            {
                "description": "Parses a docstring.",
                "properties": {
                    "arg1": {"type": "string", "description": "The first argument."}
                },
            },
            True,
        )
        self.assertEqual(parse(docstring), expected_output)

    def test_parse_without_description(self):
        docstring = """Parameters
        ----------
        arg1 : str
            The first argument.

        Returns
        -------
        dict
            A dictionary containing the description and the arguments of the function.
        """
        with self.assertRaises(ValueError):
            parse(docstring)


if __name__ == "__main__":
    unittest.main()
