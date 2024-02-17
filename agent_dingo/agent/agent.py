from typing import Callable, Union, Optional, Tuple, List, Literal
from agent_dingo.agent.parser import parse
from agent_dingo.agent.helpers import get_required_args, construct_json_repr
from agent_dingo.agent.docgen import generate_docstring
from agent_dingo.agent.function_descriptor import FunctionDescriptor
from agent_dingo.core.blocks import BaseLLM, BaseAgent, Context, ChatPrompt, KVData
from agent_dingo.core.message import UserMessage
from agent_dingo.agent.chat_context import ChatContext
from agent_dingo.agent.registry import Registry as _Registry
import json
import os
import inspect
from asyncio import run as asyncio_run, to_thread
import warnings


class Agent(BaseAgent):
    """The agent that can be used to register functions and chat with the LLM."""

    def __init__(
        self,
        llm: BaseLLM,
        max_function_calls: int = 10,
        before_function_call: Callable = None,
        allow_codegen: Union[bool, Literal["env"]] = "env",
        name="agent",
        description: str = "A helpful agent",
    ):
        if not isinstance(allow_codegen, bool) and allow_codegen != "env":
            raise ValueError(
                "allow_codegen must be a boolean or the string 'env' to use the DINGO_ALLOW_CODEGEN environment variable"
            )
        if not llm.supports_function_calls:
            raise ValueError(
                "Provided LLM does not support function calls and cannot be used with the agent."
            )
        self.model = llm
        self._allow_codegen = allow_codegen
        self._registry = _Registry()
        self.max_function_calls = max_function_calls
        self.before_function_call = before_function_call
        self.name = name
        self.description = description
        self._registered = False

    def _is_codegen_allowed(self) -> bool:
        """Determines whether docstring generation is allowed.

        Returns
        -------
        bool
            True if docstring generation is allowed, False otherwise.
        """
        if self._allow_codegen == "env":
            return bool(os.getenv("DINGO_ALLOW_CODEGEN", True))
        return self._allow_codegen

    def register_descriptor(self, descriptor: FunctionDescriptor) -> None:
        """Registers a function descriptor with the agent.

        Parameters
        ----------
        descriptor : FunctionDescriptor
            The function descriptor to register.
        """
        if descriptor.required_context_keys is not None and self._registered:
            raise ValueError(
                "required_context_keys must be None if functions are registered after the agent"
            )
        if not isinstance(descriptor, FunctionDescriptor):
            raise ValueError("descriptor must be a FunctionDescriptor")
        self._registry.add(
            name=descriptor.name,
            func=descriptor.func,
            json_repr=descriptor.json_repr,
            requires_context=descriptor.requires_context,
            required_context_keys=descriptor.required_context_keys,
        )

    def register_function(
        self, func: Callable, required_context_keys: Optional[List[str]] = None
    ) -> None:
        """Registers a function with the agent.

        Parameters
        ----------
        func : Callable
            The function to register.

        Raises
        ------
        ValueError
            Function has no docstring and code generation is not allowed
        """
        if required_context_keys is not None and self._registered:
            raise ValueError(
                "required_context_keys must be None if functions are registered after the agent"
            )
        if required_context_keys is not None:
            for key in required_context_keys:
                if not isinstance(key, str):
                    raise ValueError("required_context_keys must be a list of strings")
        docstring = func.__doc__
        if docstring is None:
            if not self._is_codegen_allowed():
                raise ValueError(
                    "Function has no docstring and code generation is not allowed"
                )
            docstring = generate_docstring(func, self.model)
        body, requires_context = parse(docstring)
        required_args = get_required_args(func)
        json_repr = construct_json_repr(
            func.__name__, body["description"], body["properties"], required_args
        )
        self._registry.add(
            func.__name__, func, json_repr, requires_context, required_context_keys
        )

    def _call_from_agent(self, query: str, chat_context: ChatContext) -> str:
        """Calls the agent from another from the agent.

        Parameters
        ----------
        query : str
            Query
        context : Context
            Chat context
        """
        prompt = ChatPrompt([UserMessage(query)])
        response = self.forward(prompt, chat_context[0], chat_context[1])
        return response["_out_0"]

    async def _async_call_from_agent(
        self, query: str, chat_context: ChatContext
    ) -> str:
        """Calls the agent from another from the agent.

        Parameters
        ----------
        query : str
            Query
        context : Context
            Chat context
        """
        prompt = ChatPrompt([UserMessage(query)])
        response = await self.async_forward(prompt, chat_context[0], chat_context[1])
        return response["_out_0"]

    def as_function_descriptor(self, as_async: bool = False) -> FunctionDescriptor:
        descriptor = FunctionDescriptor(
            name=self.name,
            func=self._call_from_agent if not as_async else self._async_call_from_agent,
            json_repr={
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The natural language query to send to the agent.",
                        }
                    },
                    "required": ["query"],
                },
            },
            requires_context=True,
            required_context_keys=self.get_required_context_keys(),
        )
        self._registered = True
        return descriptor

    def function(self, *args, **kwargs) -> Callable:
        """Registers a function with the agent and returns the function.

        Parameters
        ----------
        func : Callable
            The function to register.

        Returns
        -------
        Callable
            The function.
        """

        def outer(required_context_keys):
            def register_decorator(func):
                self.register_function(
                    func, required_context_keys=required_context_keys
                )
                return func

            return register_decorator

        if len(args) == 1 and callable(args[0]) and not kwargs:
            func = args[0]
            self.register_function(func)
            return func
        else:
            return outer(kwargs.get("required_context_keys", None))

    def get_required_context_keys(self) -> List[str]:
        # this allows to handle the case where the user registers a function after registering the agent
        return self._registry.get_required_context_keys()

    def forward(
        self, state: ChatPrompt, context: Context, store: KVData
    ) -> Tuple[str, List[dict]]:
        """Sends a message to the LLM and returns the response. Calls functions if the LLM requests it.

        Parameters
        ----------
        messages : Union[str, dict]
            The message(s) to send to the LLM
        context : ChatContext, optional
            The chat context, by default None
        Returns
        -------
        Tuple[str, List[dict]]
            A tuple containing the last response and the conversation history.
        """
        messages = state.dict
        n_calls = 0
        available_functions = self._registry.get_available_functions()
        chat_context = (context, store)
        while True:
            available_functions_i = (
                available_functions if n_calls < self.max_function_calls else None
            )
            response = self.model.send_message(
                messages,
                functions=available_functions_i,
                usage_meter=store.usage_meter,
            )
            if response.get("tool_calls"):
                messages.append(response)
                for function in response["tool_calls"]:
                    function_name = function["function"]["name"]
                    function_args = json.loads(function["function"]["arguments"])
                    f, requires_context = self._registry.get_function(function_name)
                    if requires_context:
                        function_args["chat_context"] = chat_context
                    if self.before_function_call:
                        f, function_args = self.before_function_call(
                            function_name, f, function_args
                        )
                    if inspect.iscoroutinefunction(f):
                        warnings.warn("Async function is called from a sync agent.")
                        result = asyncio_run(f(**function_args))
                    else:
                        result = f(**function_args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": function["id"],
                            "content": result,
                        }
                    )
                    n_calls += 1
            else:
                messages.append(response)
                return KVData(_out_0=response["content"])

    async def async_forward(
        self, state: ChatPrompt, context: Context, store: KVData
    ) -> Tuple[str, List[dict]]:
        """Sends a message to the LLM and returns the response. Calls functions if the LLM requests it.

        Parameters
        ----------
        messages : Union[str, dict]
            The message(s) to send to the LLM
        context : ChatContext, optional
            The chat context, by default None
        Returns
        -------
        Tuple[str, List[dict]]
            A tuple containing the last response and the conversation history.
        """
        messages = state.dict
        n_calls = 0
        available_functions = self._registry.get_available_functions()
        chat_context = (context, store)
        while True:
            available_functions_i = (
                available_functions if n_calls < self.max_function_calls else None
            )
            response = await self.model.async_send_message(
                messages,
                functions=available_functions_i,
                usage_meter=store.usage_meter,
            )
            if response.get("tool_calls"):
                messages.append(response)
                for function in response["tool_calls"]:
                    function_name = function["function"]["name"]
                    function_args = json.loads(function["function"]["arguments"])
                    f, requires_context = self._registry.get_function(function_name)
                    if requires_context:
                        function_args["chat_context"] = chat_context
                    if self.before_function_call:
                        f, function_args = self.before_function_call(
                            function_name, f, function_args
                        )
                    if inspect.iscoroutinefunction(f):
                        result = await f(**function_args)
                    else:
                        warnings.warn("Sync function is called from an async agent.")
                        result = await to_thread(f, **function_args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": function["id"],
                            "content": result,
                        }
                    )
                    n_calls += 1
            else:
                messages.append(response)
                return KVData(_out_0=response["content"])
