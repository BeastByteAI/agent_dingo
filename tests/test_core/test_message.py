import unittest
from agent_dingo.core.message import (
    Message,
    UserMessage,
    SystemMessage,
    AssistantMessage,
)


class TestMessage(unittest.TestCase):
    def test_message(self):
        m = Message("Hello")
        self.assertEqual(m.content, "Hello")
        self.assertEqual(m.role, "undefined")
        self.assertEqual(m.dict, {"role": "undefined", "content": "Hello"})

    def test_user_message(self):
        m = UserMessage("Hello")
        self.assertEqual(m.content, "Hello")
        self.assertEqual(m.role, "user")
        self.assertEqual(m.dict, {"role": "user", "content": "Hello"})

    def test_system_message(self):
        m = SystemMessage("Hello")
        self.assertEqual(m.content, "Hello")
        self.assertEqual(m.role, "system")
        self.assertEqual(m.dict, {"role": "system", "content": "Hello"})

    def test_assistant_message(self):
        m = AssistantMessage("Hello")
        self.assertEqual(m.content, "Hello")
        self.assertEqual(m.role, "assistant")
        self.assertEqual(m.dict, {"role": "assistant", "content": "Hello"})


if __name__ == "__main__":
    unittest.main()
