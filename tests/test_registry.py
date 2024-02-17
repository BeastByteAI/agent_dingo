import unittest
from agent_dingo.agent.registry import Registry as _Registry


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = _Registry()

    def test_add_function(self):
        def func():
            pass

        json_repr = {"name": "func"}
        self.registry.add("func", func, json_repr, False)
        self.assertEqual(len(self.registry._Registry__functions), 1)

    def test_get_function(self):
        def func():
            pass

        json_repr = {"name": "func"}
        self.registry.add("func", func, json_repr, False)
        func, requires_context = self.registry.get_function("func")
        self.assertEqual(callable(func), True)
        self.assertEqual(requires_context, False)

    def test_get_available_functions(self):
        def func1():
            pass

        def func2():
            pass

        json_repr1 = {"name": "func1"}
        json_repr2 = {"name": "func2"}
        self.registry.add("func1", func1, json_repr1, False)
        self.registry.add(
            "func2", func2, json_repr2, True, required_context_keys=["any"]
        )
        available_functions = self.registry.get_available_functions()
        self.assertEqual(len(available_functions), 2)
        self.assertEqual(available_functions[0]["name"], "func1")
        self.assertEqual(available_functions[1]["name"], "func2")


if __name__ == "__main__":
    unittest.main()
