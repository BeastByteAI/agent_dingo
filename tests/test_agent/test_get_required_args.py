import unittest
from agent_dingo.agent.helpers import get_required_args
from agent_dingo.agent.chat_context import ChatContext


class TestGetRequiredArgs(unittest.TestCase):
    def test_no_args(self):
        def func():
            pass

        self.assertEqual(get_required_args(func), [])

    def test_required_args(self):
        def func(a, b, c):
            pass

        self.assertEqual(get_required_args(func), ["a", "b", "c"])

    def test_optional_args(self):
        def func(a, b, c=None):
            pass

        self.assertEqual(get_required_args(func), ["a", "b"])

    def test_mixed_args(self):
        def func(a, b, c=None, d=None):
            pass

        self.assertEqual(get_required_args(func), ["a", "b"])

    def test_with_chat_context(self):
        def func(a, b, chat_context: ChatContext):
            pass

        self.assertEqual(get_required_args(func), ["a", "b"])

    def test_wrong_chat_context_type(self):
        def func(a, b, chat_context: str):
            pass

        self.assertEqual(get_required_args(func), ["a", "b", "chat_context"])

    def test_wrong_chat_context_name(self):
        def func(a, b, chat_context_: ChatContext):
            pass

        self.assertEqual(get_required_args(func), ["a", "b", "chat_context_"])


if __name__ == "__main__":
    unittest.main()
