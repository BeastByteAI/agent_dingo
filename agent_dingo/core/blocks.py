from __future__ import annotations
from typing import Any, Coroutine, Optional, Union, List, Dict
from abc import ABC, abstractmethod
from agent_dingo.core.message import Message
from agent_dingo.core.state import State, ChatPrompt, KVData, Context, Store, UsageMeter
from agent_dingo.core.output_parser import BaseOutputParser, DefaultOutputParser
import re
import joblib
import inspect
import warnings


import os

if os.environ.get("DINGO_ALLOW_NESTED_ASYNCIO", False):
    import nest_asyncio

    nest_asyncio.apply()

from asyncio import (
    to_thread,
    gather,
    run as asyncio_run,
)


class Block(ABC):
    """Base building block of a pipeline"""

    @abstractmethod
    def forward(self, state: Optional[State], context: Context, store: Store) -> State:
        pass

    async def async_forward(
        self, state: Optional[State], context: Context, store: Store
    ) -> State:
        # ideally, this is never called as all children should implement async forward
        warnings.warn(
            f"Called async_forward, but {self.__class__} block does not have an async implementation."
        )
        return await to_thread(self.forward, state, context, store)

    @abstractmethod
    def get_required_context_keys(self) -> List[str]:
        """Each block must specify the keys it requires from the context."""
        pass

    def __rshift__(self, other: Block) -> Pipeline:
        return Pipeline() >> self >> other

    def __lshift__(self, other: Block) -> Pipeline:
        if isinstance(other, Pipeline):
            other.add_block(self)
            return other
        return Pipeline() >> other >> self

    def as_pipeline(self) -> Pipeline:
        return Pipeline() >> self

    def __and__(self, other: Block) -> Parallel:
        return Parallel() & self & other


######### REASONERS #########


class BaseReasoner(Block):
    """A reasoner is a block that takes a prompt and returns a KVData object."""

    @abstractmethod
    def forward(self, state: ChatPrompt, context: Context, store: Store) -> KVData:
        pass


class BaseLLM(BaseReasoner):
    """LLM is a type of reasoner that directly interacts with a language model."""

    supports_function_calls = False

    @abstractmethod
    def send_message(
        self, messages, functions=None, usage_meter: UsageMeter = None, **kwargs
    ):
        pass

    @abstractmethod
    async def async_send_message(
        self, messages, functions=None, usage_meter: UsageMeter = None, **kwargs
    ):
        pass

    def forward(self, state: ChatPrompt, context: Context, store: Store) -> KVData:
        if not isinstance(state, ChatPrompt):
            raise TypeError(f"State must be a ChatPrompt, got {type(state)}")
        new_state = KVData(
            _out_0=self.process_prompt(state, usage_meter=store.usage_meter)
        )
        return new_state

    async def async_forward(self, state: State | None, context: Context, store: Store):
        if not isinstance(state, ChatPrompt):
            raise TypeError(f"State must be a ChatPrompt, got {type(state)}")
        new_state = KVData(
            _out_0=await self.async_process_prompt(state, usage_meter=store.usage_meter)
        )
        return new_state

    def __call__(self, state: ChatPrompt) -> str:
        return self.process_prompt(state)

    def get_required_context_keys(self) -> List[str]:
        return []

    def process_prompt(
        self, prompt: ChatPrompt, usage_meter: Optional[UsageMeter] = None, **kwargs
    ):
        return self.send_message(prompt.dict, None, usage_meter)["content"]

    async def async_process_prompt(
        self, prompt: ChatPrompt, usage_meter: Optional[UsageMeter] = None, **kwargs
    ):
        return (await self.async_send_message(prompt.dict, None, usage_meter))[
            "content"
        ]


class BaseAgent(BaseReasoner):
    """An agent is a type of reasoner that can autonomously perform multi-step reasoning."""

    pass


######### KVData Processors #########


class BaseKVDataProcessor(Block):
    """KVDataProcessor is a block that takes a KVData object and returns a KVData object."""

    @abstractmethod
    def forward(self, state: KVData, context: Context, store: Store) -> KVData:
        pass


class Squash(BaseKVDataProcessor):
    def __init__(self, template: str):
        """Squash block takes a KVData with multiple keys and squashes them into a single key using a template string.

        Parameters
        ----------
        template : str
            Template string with (in-order) placeholders for each key in the KVData object.
        """
        self.template = template

    def forward(self, state: KVData, context: Context, store: Store) -> KVData:
        return KVData(_out_0=self.template.format(*state.values()))

    async def async_forward(self, state: KVData, context: Context, store: Store):
        return self.forward(state, context, store)

    def get_required_context_keys(self) -> List[str]:
        return []


######### Prompt Builders #########


class BasePromptBuilder(Block):
    """PromptBuilder is a block that takes a KVData object (or None) and returns a ChatPrompt."""

    @abstractmethod
    def forward(
        self, state: Optional[KVData], context: Context, store: Store
    ) -> ChatPrompt:
        pass


class PromptBuilder(BasePromptBuilder):
    def __init__(
        self,
        messages: list[Message],
        from_state: Optional[Union[List[str], Dict[str, str]]] = None,
        from_store: Optional[Union[List[str], Dict[str, str]]] = None,
    ):
        """
        PromptBuilder formats the list of messages (templates) with values from the state, store and context.

        Parameters
        ----------
        messages : list[Message]
            List of message templates to format.
        from_state : Optional[Union[List[str], Dict[str, str]]], optional
            List of strings or mapping template->state that defines which placeholders should be populated by state values, by default None
        from_store : Optional[Union[List[str], Dict[str, str]]], optional
             List of strings or mapping template->store (where the store key is formated as <outer_key>.<inner_key>) that defines which placeholders should be populated by state values, by default None
        """
        self.messages = messages
        self._from_state_keys = []
        self._from_store_keys = []
        if from_state is None:
            self._from_state = {}
        elif isinstance(from_state, list):
            self._from_state = {}
            self._from_state_keys.extend(from_state)
            for i, k in enumerate(from_state):
                self._from_state[k] = f"_out_{i}"
        elif isinstance(from_state, dict):
            self._from_state = from_state
            self._from_state_keys.extend(from_state.keys())
        else:
            raise TypeError(
                f"from_state must be a list or dict, got {type(from_state)}"
            )
        if from_store is None:
            self._from_store = {}
        elif isinstance(from_store, list):
            self._from_store = {}
            self._from_store_keys.extend(from_store)
            for i, k in enumerate(from_store):
                self._from_store[k] = f"_out_{i}"
        elif isinstance(from_store, dict):
            self._from_store = from_store
            self._from_store_keys = from_store.keys()
        else:
            raise TypeError(
                f"from_store must be a list or dict, got {type(from_store)}"
            )

        self._placeholder_names = self._get_placeholder_names()

    def _get_placeholder_names(self) -> set:
        placeholder_pattern = r"\{(\w+)\}"
        placeholder_names = set()

        for message in self.messages:
            found_placeholders = re.findall(placeholder_pattern, message.content)
            placeholder_names.update(found_placeholders)

        return placeholder_names

    def forward(
        self, state: Optional[KVData], context: Context, store: Store
    ) -> ChatPrompt:
        values = {}
        for n in self._placeholder_names:
            if n in self._from_state.keys():
                values[n] = state[self._from_state[n]]
            elif n in self._from_store.keys():
                if "." in self._from_store[n]:
                    outer, inner = self._from_store[n].split(".")
                    values[n] = store.get_data(outer)[inner]
                else:
                    raise ValueError(
                        "Store key must be formatted as <outer_key>.<inner_key>"
                    )
            elif n in context.keys():
                values[n] = context[n]
            else:
                raise KeyError(f"Could not find value for placeholder {n}")
        updated_messages = [type(m)(m.content.format(**values)) for m in self.messages]
        return ChatPrompt(updated_messages)

    async def async_forward(self, state: KVData, context: Context, store: Store):
        return self.forward(state, context, store)

    def get_required_context_keys(self) -> List[str]:
        keys = []
        for n in self._placeholder_names:
            if n not in self._from_state_keys and n not in self._from_store_keys:
                keys.append(n)
        return keys


######### Prompt Modifiers #########


class BasePromptModifier(Block):
    """A prompt modifier is a block that takes a ChatPrompt and returns a ChatPrompt."""

    @abstractmethod
    def forward(self, state: ChatPrompt, context: Context, store: Store) -> ChatPrompt:
        pass


######### Special Blocks #########


class Pipeline(Block):
    def __init__(self, output_parser: Optional[BaseOutputParser] = None):
        """
        A pipeline is a sequence of blocks that are executed in order.
        The pipeline itself is a block that can be used in other pipelines.

        Parameters
        ----------
        output_parser : Optional[BaseOutputParser], optional
            custom output parser of the last step, by default None
        """
        self.output_parser: BaseOutputParser = output_parser or DefaultOutputParser()
        self._blocks = []

    def add_block(self, block: Block):
        """
        Add a block to the pipeline.

        Parameters
        ----------
        block : Block
            Block to add to the pipeline.
        """
        if not isinstance(block, Block):
            raise TypeError(f"Expected a Block, got {type(block)}")
        self._blocks.append(block)

    def forward(self, state: Optional[State], context: Context, store: Store) -> State:
        running_state = state
        for block in self._blocks:
            running_state = block.forward(
                state=running_state, context=context, store=store
            )
        return running_state

    async def async_forward(
        self, state: Optional[State], context: Context, store: Store
    ) -> State:
        running_state = state
        for block in self._blocks:
            running_state = await block.async_forward(
                state=running_state, context=context, store=store
            )
        return running_state

    def run(self, _state: Optional[State] = None, **kwargs: Dict[str, str]):
        """
        Runs the pipeline with the given state and context (populated with kwargs).
        Each run initializes a new empty store.
        The output of the last block is parsed using the output_parser and returned.

        Parameters
        ----------
        _state : Optional[State], optional
            initial state, by default None
        """
        context = Context(**kwargs)
        store = Store()
        out = self.forward(state=_state, context=context, store=store)
        return self.output_parser.parse(out), store.usage_meter.get_usage()

    async def async_run(
        self, _state: Optional[State] = None, **kwargs: Dict[str, str]
    ) -> str:
        context = Context(**kwargs)
        store = Store()
        out = await self.async_forward(state=_state, context=context, store=store)
        return self.output_parser.parse(out), store.usage_meter.get_usage()

    def __rshift__(self, other: Block) -> Pipeline:
        self.add_block(other)
        return self

    def get_required_context_keys(self) -> List[str]:
        keys = []
        for block in self._blocks:
            keys.extend(block.get_required_context_keys())
        return keys


class Parallel(Block):
    def __init__(self):
        """
        A parallel block executes multiple sub-blocks in parallel. The output of each block is stored as a separate key in the KVData object.
        """
        self.blocks = []

    def add_block(self, block: Block):
        """
        Add a block to the parallel block.

        Parameters
        ----------
        block : Block
            Block to add.
        """
        if not isinstance(block, Block):
            raise TypeError(f"Expected a Block, got {type(block)}")
        self.blocks.append(block)

    def forward(self, state: Optional[State], context: Context, store: Store) -> State:
        # run all blocks in parallel
        states = joblib.Parallel(n_jobs=len(self.blocks), backend="threading")(
            joblib.delayed(block.forward)(state=state, context=context, store=store)
            for block in self.blocks
        )
        out = {}
        for i, state in enumerate(states):
            if i == 0 and isinstance(state, ChatPrompt):
                # allow a special case where the first block returns a ChatPrompt
                # the ouput of remaining branches will be ignored
                return state
            if not isinstance(state, KVData):
                raise TypeError(
                    f"Expected KVData, got {type(state)} from block {i} of {len(states)}"
                )
            if len(state.keys()) != 1:
                raise ValueError(
                    f"Expected KVData with one key `_out_0`, got {len(state.keys())} keys from block {i} of {len(states)}"
                )
            out[f"_out_{i}"] = state["_out_0"]
        return KVData(**out)

    async def async_forward(
        self, state: Optional[State], context: Context, store: Store
    ) -> State:
        tasks = [block.async_forward(state, context, store) for block in self.blocks]
        states = await gather(*tasks)

        out = {}
        for i, state in enumerate(states):
            if i == 0 and isinstance(state, ChatPrompt):
                return state
            if not isinstance(state, KVData):
                raise TypeError(
                    f"Expected KVData, got {type(state)} from block {i} of {len(states)}"
                )
            if len(state.keys()) != 1:
                raise ValueError(
                    f"Expected KVData with one key `_out_0`, got {len(state.keys())} keys from block {i} of {len(states)}"
                )
            out[f"_out_{i}"] = state["_out_0"]
        return KVData(**out)

    def __and__(self, other: Block) -> Parallel:
        self.add_block(other)
        return self

    def get_required_context_keys(self) -> List[str]:
        keys = []
        for block in self.blocks:
            keys.extend(block.get_required_context_keys())


class Identity(Block):
    """NO-OP block that returns the input state as is."""

    def forward(self, state: Optional[State], context: Context, store: Store) -> State:
        return state

    async def async_forward(
        self, state: Optional[State], context: Context, store: Store
    ) -> State:
        return state

    def get_required_context_keys(self) -> List[str]:
        return []


class SaveState(Block):
    def __init__(self, key: str):
        """Saves the current state to the store.

        Parameters
        ----------
        key : str
            Key to save the state under.
        """
        self.key = key

    def get_required_context_keys(self) -> List[str]:
        return []

    def forward(self, state: State | None, context: Context, store: Store) -> State:
        store.update(self.key, state)
        return state

    async def async_forward(
        self, state: State | None, context: Context, store: Store
    ) -> State:
        return self.forward(state, context, store)


class LoadState(Block):
    def __init__(self, from_: str, key: str):
        """
        Loads the state from the store.

        Parameters
        ----------
        from_ : str
            Defines whether to load from a Prompt of KVData section of the store.
        key : str
            Key to load the state from.
        """
        if from_ not in ["prompts", "data"]:
            raise ValueError(f"from_ must be 'store' or 'context', got {from_}")
        self.from_ = from_
        self.key = key

    def get_required_context_keys(self) -> List[str]:
        return []

    def forward(self, state: State | None, context: Context, store: Store) -> State:
        if self.from_ == "prompts":
            return store.get_prompt(self.key)
        elif self.from_ == "data":
            return store.get_data(self.key)
        else:
            raise ValueError(f"from_ must be 'store' or 'context', got {self.from_}")

    async def async_forward(
        self, state: State | None, context: Context, store: Store
    ) -> State:
        return self.forward(state, context, store)


class InlineBlock(Block):
    def __init__(self, required_context_keys: Optional[List[str]] = None):
        """A decorator to convert a function into an inline block.

        Parameters
        ----------
        required_context_keys : Optional[List[str]], optional
            specifies the context keys required by the function, by default None
        """
        self.required_context_keys = required_context_keys or []
        self.func = None

    def get_required_context_keys(self) -> List[str]:
        return self.required_context_keys

    def __call__(self, func):
        self.func = func
        return self

    def _get_output(self, out) -> State:
        if isinstance(out, State.__args__):
            return out
        elif isinstance(out, dict):
            return KVData(**out)
        elif isinstance(out, str):
            return KVData(_out_0=out)
        elif isinstance(out, (list, tuple)):
            return KVData(**{f"_out_{i}": v for i, v in enumerate(out)})
        raise TypeError(f"Expected a State, dict, str, or list, got {type(out)}")

    def forward(self, state: State | None, context: Context, store: Store) -> State:
        if inspect.iscoroutinefunction(self.func):
            warnings.warn(f"Called forward on an async inline block.")
            out = asyncio_run(self.func(state, context, store))
        else:
            out = self.func(state, context, store)
        return self._get_output(out)

    async def async_forward(
        self, state: State | None, context: Context, store: Store
    ) -> State:
        if inspect.iscoroutinefunction(self.func):
            out = await self.func(state, context, store)
        else:
            warnings.warn(f"Called async_forward on a non-async inline block.")
            out = await to_thread(self.func, state, context, store)
        return self._get_output(out)
