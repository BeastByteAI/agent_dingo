import unittest
from agent_dingo.core.output_parser import BaseOutputParser, DefaultOutputParser
from agent_dingo.core.state import State, ChatPrompt, KVData
from agent_dingo.core.message import Message


class TestOutputParser(unittest.TestCase):
    def test_base_output_parser(self):
        class TestOutputParser(BaseOutputParser):
            def _parse_chat(self, output: ChatPrompt) -> str:
                return "chat"

            def _parse_kvdata(self, output: KVData) -> str:
                return "kvdata"

        parser = TestOutputParser()
        self.assertEqual(parser.parse(KVData(a="1")), "kvdata")
        self.assertEqual(parser.parse(ChatPrompt([Message("Hello")])), "chat")
        with self.assertRaises(TypeError):
            parser.parse("invalid")

    def test_default_output_parser(self):
        parser = DefaultOutputParser()
        with self.assertRaises(RuntimeError):
            parser.parse(ChatPrompt([Message("Hello")]))
        self.assertEqual(parser.parse(KVData(_out_0="output")), "output")
        with self.assertRaises(KeyError):
            parser.parse(KVData(a="1"))


if __name__ == "__main__":
    unittest.main()
