import unittest
from agent_dingo.core.state import ChatPrompt, KVData, Context, UsageMeter, Store
from agent_dingo.core.message import Message


class TestState(unittest.TestCase):
    def test_chat_prompt(self):
        cp = ChatPrompt([Message("Hello")])
        self.assertEqual(cp.dict, [{"role": "undefined", "content": "Hello"}])

    def test_kvdata(self):
        kv = KVData(a="1", b="2")
        self.assertEqual(kv["a"], "1")
        self.assertEqual(kv["b"], "2")
        self.assertEqual(kv.dict, {"a": "1", "b": "2"})
        with self.assertRaises(KeyError):
            kv.update("a", "3")

    def test_context(self):
        ctx = Context(a="1", b="2")
        with self.assertRaises(RuntimeError):
            ctx.update("a", "3")

    def test_usage_meter(self):
        um = UsageMeter()
        um.increment(10, 20)
        self.assertEqual(
            um.get_usage(),
            {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

    def test_store(self):
        st = Store()
        st.update("data", KVData(a="1"))
        st.update("prompt", ChatPrompt([Message("Hello")]))
        st.update("misc", "misc")
        self.assertEqual(st.get_data("data").dict, {"a": "1"})
        self.assertEqual(
            st.get_prompt("prompt").dict, [{"role": "undefined", "content": "Hello"}]
        )
        self.assertEqual(st.get_misc("misc"), "misc")


if __name__ == "__main__":
    unittest.main()
