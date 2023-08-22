from typing import Callable, Union, Optional, Tuple, List, Literal
from agent_dingo.parser import parse
from agent_dingo.helpers import get_required_args, construct_json_repr
from agent_dingo.context import ChatContext
from agent_dingo.chat import send_message
from agent_dingo.docgen import generate_docstring
from agent_dingo.usage import UsageMeter
from agent_dingo.function_descriptor import FunctionDescriptor
from dataclasses import asdict
import json
import os


class _Registry:
    """A registry for functions that can be called by the agent."""

    def __init__(self):
        self.__functions = {}

    def add(
        self, name: str, func: Callable, json_repr: dict, requires_context: bool
    ) -> None:
        """Adds a function to the registry.

        Parameters
        ----------
        name : str
            The name of the function.
        func : Callable
            The function.
        json_repr : dict
            The JSON representation of the function to be provided to the LLM.
        requires_context : bool
            Indicates whether the function requires a ChatContext object as one of its arguments.
        """
        self.__functions[name] = {
            "func": func,
            "json_repr": json_repr,
            "requires_context": requires_context,
        }

    def get_function(self, name: str) -> Tuple[Callable, bool]:
        """Retrieves a function from the registry.

        Parameters
        ----------
        name : str
            The name of the function.

        Returns
        -------
        Tuple[Callable, bool]
            A tuple containing the function and a boolean indicating whether the function requires a ChatContext object as one of its arguments.
        """
        return (
            self.__functions[name]["func"],
            self.__functions[name]["requires_context"],
        )

    def get_available_functions(self) -> List[dict]:
        """Returns a list of JSON representations of the functions in the registry.

        Returns
        -------
        List[dict]
            A list of JSON representations of the functions in the registry.
        """
        return [self.__functions[name]["json_repr"] for name in self.__functions]


class AgentDingo:
    """The agent that can be used to register functions and chat with the LLM."""

    def __init__(self, allow_codegen: Union[bool, Literal["env"]] = "env"):
        """
        Parameters
        ----------
        allow_codegen : Union[bool, Literal["env"]], optional
            Determines whether docstring generation is allowed, by default "env"

        Raises
        ------
        ValueError
            Invalid value for allow_codegen.
        """
        if not isinstance(allow_codegen, bool) and allow_codegen != "env":
            raise ValueError(
                "allow_codegen must be a boolean or the string 'env' to use the DINGO_ALLOW_CODEGEN environment variable"
            )
        self._allow_codegen = allow_codegen
        self._registry = _Registry()

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
        if not isinstance(descriptor, FunctionDescriptor):
            raise ValueError("descriptor must be a FunctionDescriptor")
        self._registry.add(
            name=descriptor.name,
            func=descriptor.func,
            json_repr=descriptor.json_repr,
            requires_context=descriptor.requires_context,
        )

    def register_function(self, func: Callable) -> None:
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
        docstring = func.__doc__
        if docstring is None:
            if not self._is_codegen_allowed():
                raise ValueError(
                    "Function has no docstring and code generation is not allowed"
                )
            docstring = generate_docstring(
                func, os.environ.get("DINGO_CODEGEN_MODEL", "gpt-3.5-turbo-0613")
            )
        body, requires_context = parse(docstring)
        required_args = get_required_args(func)
        json_repr = construct_json_repr(
            func.__name__, body["description"], body["properties"], required_args
        )
        self._registry.add(func.__name__, func, json_repr, requires_context)

    def function(self, func: Callable) -> Callable:
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
        self.register_function(func)
        return func

    def chat(
        self,
        messages: Union[str, dict],
        chat_context: Optional[ChatContext] = None,
        model: str = "gpt-3.5-turbo-0613",
        temperature: float = 1.0,
        max_function_calls: int = 10,
        before_function_call: Callable = None,
        usage_meter: Optional[UsageMeter] = None,
    ) -> Tuple[str, List[dict]]:
        """Sends a message to the LLM and returns the response. Calls functions if the LLM requests it.

        Parameters
        ----------
        messages : Union[str, dict]
            The message(s) to send to the LLM
        chat_context : ChatContext, optional
            The chat context, by default None
        model : str, optional
            The model to be used, by default "gpt-3.5-turbo-0613"
        temperature : float, optional
            The temperature to be used, by default 1.0
        max_function_calls : int, optional
            The maximum number of function calls to be made, by default 10
        before_function_call : Callable, optional
            A function to be called before a function call is made, by default None

        Returns
        -------
        Tuple[str, List[dict]]
            A tuple containing the last response and the conversation history.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        n_calls = 0
        available_functions = self._registry.get_available_functions()
        if not chat_context:
            chat_context = ChatContext()
        while True:
            available_functions_i = (
                available_functions if n_calls < max_function_calls else None
            )
            response = send_message(
                messages,
                model=model,
                functions=available_functions_i,
                temperature=temperature,
                usage_meter=usage_meter,
            )
            if response.get("function_call"):
                function_name = response["function_call"]["name"]
                function_args = json.loads(response["function_call"]["arguments"])
                f, requires_context = self._registry.get_function(function_name)
                if requires_context:
                    function_args["chat_context"] = chat_context
                if before_function_call:
                    f, function_args = before_function_call(
                        function_name, f, function_args
                    )
                result = f(**function_args)
                messages.append(response)
                messages.append(
                    {
                        "role": "function",
                        "name": function_name,
                        "content": result,
                    }
                )
                n_calls += 1
            else:
                messages.append(response)
                return response["content"], messages
