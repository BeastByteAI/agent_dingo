import unittest
from agent_dingo.core.blocks import (
    Squash,
    PromptBuilder,
    Pipeline,
    Parallel,
    Identity,
    SaveState,
    LoadState,
    InlineBlock,
)
from agent_dingo.core.state import State, ChatPrompt, KVData, Context, Store, UsageMeter
from agent_dingo.core.message import Message


class TestBlocks(unittest.TestCase):
    def test_squash(self):
        s = Squash("{0} {1}")
        state = KVData(_out_0="Hello", _out_1="World")
        context = Context()
        store = Store()
        self.assertEqual(s.forward(state, context, store)["_out_0"], "Hello World")

    def test_prompt_builder(self):
        pb = PromptBuilder([Message("Hello {name}")], from_state=["_out_0"])
        state = KVData(_out_0="World")
        context = Context(name="World")
        store = Store()
        self.assertEqual(
            pb.forward(state, context, store).messages[0].content, "Hello World"
        )

    def test_pipeline(self):
        p = Pipeline()
        p.add_block(Squash("{0} {1}"))
        state = KVData(_out_0="Hello", _out_1="World")
        context = Context()
        store = Store()
        self.assertEqual(p.forward(state, context, store)["_out_0"], "Hello World")
        self.assertEqual(p.run(state)[0], "Hello World")

    def test_pipeline_build(self):
        block1 = Identity()
        block2 = Identity()
        pipeline = block1 >> block2
        self.assertIsInstance(pipeline, Pipeline)
        self.assertEqual(len(pipeline._blocks), 2)
        self.assertIs(pipeline._blocks[0], block1)
        self.assertIs(pipeline._blocks[1], block2)

    def test_pipeline_build_with_parallel(self):
        block1 = Identity()
        block2 = Identity()
        squash = Squash("{0} {1}")
        pipeline = (block1 & block2) >> squash
        self.assertIsInstance(pipeline, Pipeline)
        self.assertEqual(len(pipeline._blocks), 2)
        self.assertIsInstance(pipeline._blocks[0], Parallel)
        self.assertIs(pipeline._blocks[1], squash)

    def test_identity(self):
        i = Identity()
        state = KVData(_out_0="Hello")
        context = Context()
        store = Store()
        self.assertIs(i.forward(state, context, store), state)

    def test_save_state(self):
        ss = SaveState("key")
        state = KVData(_out_0="Hello")
        context = Context()
        store = Store()
        self.assertEqual(ss.forward(state, context, store)["_out_0"], "Hello")
        self.assertIs(store.get_data("key"), state)

    def test_load_state(self):
        ls = LoadState("data", "key")
        state = KVData(_out_0="Hello")
        context = Context()
        store = Store()
        store.update("key", state)
        self.assertIs(ls.forward(state, context, store), state)

    def test_inline_block(self):
        ib = InlineBlock()
        state_ = KVData(_out_0="Hello")

        @ib
        def func(state, context, store):
            return state_

        state = KVData(_out_0="World")
        context = Context()
        store = Store()
        self.assertIs(func.forward(state, context, store), state_)


if __name__ == "__main__":
    unittest.main()
