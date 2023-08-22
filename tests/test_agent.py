import unittest
from unittest.mock import patch
from agent_dingo.agent import AgentDingo
from agent_dingo.function_descriptor import FunctionDescriptor


class TestAgentDingo(unittest.TestCase):
    def setUp(self):
        self.agent = AgentDingo()

    def test_register_function(self):
        def func(arg: str):
            """_summary_

            Parameters
            ----------
            arg : str
                _description_
            """
            pass

        self.agent.register_function(func)
        self.assertEqual(len(self.agent._registry._Registry__functions), 1)

    def test_register_descriptor(self):
        d = FunctionDescriptor(
            name="function_from_descriptor",
            func=lambda arg: None,
            json_repr={},
            requires_context=False,
        )
        self.agent.register_descriptor(d)
        self.assertEqual(len(self.agent._registry._Registry__functions), 1)
        self.assertIn("function_from_descriptor", self.agent._registry._Registry__functions.keys())

    def test_function_decorator(self):
        @self.agent.function
        def func(arg: str):
            """_summary_

            Parameters
            ----------
            arg : str
                _description_
            """
            pass

        self.assertEqual(len(self.agent._registry._Registry__functions), 1)

    @patch(
        "agent_dingo.agent.send_message",
        return_value=({"role": "assistant", "content": "Hello, world!"}),
    )
    def test_chat(self, mock_send_message):
        @self.agent.function
        def func(arg: str):
            """_summary_

            Parameters
            ----------
            arg : str
                _description_
            """
            pass

        messages = [{"role": "user", "content": "Say hello!"}]
        response, history = self.agent.chat(messages)
        self.assertEqual(response, "Hello, world!")
        self.assertEqual(len(history), 2)


if __name__ == "__main__":
    unittest.main()
