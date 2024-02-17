import unittest
from unittest.mock import patch
from agent_dingo.agent import Agent
from agent_dingo.agent.function_descriptor import FunctionDescriptor
from fake_llm import FakeLLM


class TestAgentDingo(unittest.TestCase):
    def setUp(self):
        llm = FakeLLM()
        self.agent = Agent(llm)

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
        self.assertIn(
            "function_from_descriptor", self.agent._registry._Registry__functions.keys()
        )

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


if __name__ == "__main__":
    unittest.main()
